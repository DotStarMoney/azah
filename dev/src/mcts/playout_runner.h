#ifndef AZAH_MCTS_PLAYOUT_RUNNER_H_
#define AZAH_MCTS_PLAYOUT_RUNNER_H_

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include "../nn/data_types.h"
#include "../nn/init.h"
#include "absl/random/random.h"
#include "Eigen/Core"
#include "game_network_work_item.h"
#include "glog/logging.h"
#include "lock_by_key.h"
#include "network_dispatch_queue.h"
#include "state_cache.h"
#include "visit_table.h"

namespace azah {
namespace mcts {
namespace internal {

template <typename GameSubclass>
struct PlayoutConfig {
  // The game state to evaluate from.
  GameSubclass game;

  // Number of playouts to run.
  int n;

  // Below are "temperatures", or interpolants between a computed and uniform
  // distribution on [0, 1]. Higher temperatures lead to greater uniformity of
  // the MCTS exploration.

  // The temperature of the outcome term.
  float outcome_temp;

  // The temperature of the policy term.
  float policy_temp;

  // The temperature of the revisit term.
  float revisit_temp;

  // The temperature of the white noise term.
  float noise_temp;
};

template <typename GameSubclass>
struct PlayoutResult {
  // The outcome order-rotated s.t. the player who's move it is to make is in
  // the first row.
  nn::Matrix<GameSubclass::players_n(), 1> outcome;

  // The search proportion across predicted options.
  nn::DynamicMatrix policy;

  // The move option with the greatest average win proportion across playouts.
  int max_option_i;
};

template <typename GameSubclass, typename GameNetworkSubclass, int Shards, 
          int CacheBlocks, int CacheRowsPerBlock>
class PlayoutRunner {
 private:
  // We combine the outcome estimation and policy together in each cache row.
  static constexpr int kCacheValuesPerRow =
      GameSubclass::players_n() + GameSubclass::max_move_options_cache_hint();

  using Cache = StateCache<std::string, float, CacheBlocks, CacheRowsPerBlock,
                           kCacheValuesPerRow>;
  using TempCacheKey = StateCache<std::string, float, CacheBlocks,
                                  CacheRowsPerBlock, 
                                  kCacheValuesPerRow>::TempKey;
 public:
  PlayoutRunner(const PlayoutRunner&) = delete;
  PlayoutRunner& operator=(const PlayoutRunner&) = delete;

  PlayoutRunner() {}

  void ClearModelOutputCache() {
    cache_.Clear();
  }

  PlayoutResult<GameSubclass> Playout(
      const PlayoutConfig<GameSubclass>& config,
      NetworkDispatchQueue<GameNetworkSubclass>& work_queue) {
    work_queue.Drain();

    std::vector<std::unique_ptr<PlayoutState>> playout_states;
    for (int i = 0; i < config.n; ++i) {
      playout_states.push_back(
          std::make_unique<PlayoutState>(config, config.game));
    }

    for (auto& state_ptr : playout_states) {
      work_queue.AddWork(std::make_unique<FanoutWorkElement>(
          config, *state_ptr, *this, work_queue));
    }
    work_queue.Drain();
    visit_table_.Clear();

    PlayoutResult<GameSubclass> result;

    result.outcome = nn::init::Zeros<GameSubclass::players_n(), 1>();
    // The accumulated chance that the player who's turn it currently is wins
    // after making one of the possible moves. We'll later divide this by the
    // number of visits to each move and select the highest one.
    std::vector<float> move_totals(config.game.CurrentMovesN(), 0);
    // The number of times each move at the root state was visited across
    // playouts.
    std::vector<int> moves_searched(config.game.CurrentMovesN(), 0);
    for (auto& state_ptr : playout_states) {
      move_totals[state_ptr->move_index] += state_ptr->outcome[
          config.game.CurrentPlayerI()];
      ++(moves_searched[state_ptr->move_index]);

      result.outcome = (result.outcome.array()
          + Eigen::Map<nn::Matrix<GameSubclass::players_n(), 1>>(
              state_ptr->outcome.data()).array()).matrix();
    }
    result.outcome /= static_cast<float>(config.n);
    // Rotate the outcome matrix data to move the player who's turn it currently
    // is into the first row.
    std::rotate(
        result.outcome.data(),
        result.outcome.data() + config.game.CurrentPlayerI(),
        result.outcome.data() + GameSubclass::players_n());

    // Set the desired policy to be the averaged outcomes across the searched
    // moves.
    int move_index = 0;
    result.policy = config.game.PolicyMask();
    for (int i = 0; i < result.policy.rows(); ++i) {
      if (result.policy(i, 0) == 0.0f) continue;
      if (move_totals[move_index] > 0.0f) {
        move_totals[move_index] /= 
            static_cast<float>(moves_searched[move_index]);
        result.policy(i, 0) = move_totals[move_index];
      }
      ++move_index;
    }
    // The policy should be a proportion across averaged outcomes.
    result.policy /= result.policy.sum();

    result.max_option_i = std::distance(
        move_totals.begin(), 
        std::max_element(move_totals.begin(), move_totals.end()));

    return result;
  }

 private:
  Cache cache_;
  VisitTable<std::string, Shards> visit_table_;
  LockByKey<std::string, Shards> state_lock_;

  struct PlayoutState {
    PlayoutState(const PlayoutConfig<GameSubclass>& config, 
                 const GameSubclass& game) : 
        config(config), move_index(-1), evals_remaining(-1), on_root(true), 
        game(game) {}

    const PlayoutConfig<GameSubclass>& config;

    // The playout's outcome.
    std::array<float, GameSubclass::players_n()> outcome;

    // Index of the move made after fanning out the root for this playout.
    int move_index;

    // Whether or not the playout is currently evaluating the root.
    bool on_root;

    // The state of the game currently being evaluated in this playout.
    GameSubclass game;

    // The results of evaluating each move in the current game state.
    //
    // Each entry is a tuple of the sub-game state_uid, and the predicted
    // outcome for the player who's move it is to make at that sub-game.
    std::vector<std::tuple<std::string, float>> eval_results;

    // Number of completed evaluations in the current fanout for this playout.
    std::atomic<uint32_t> evals_remaining;
  };

  class PlayoutWorkElement : public GameNetworkWorkItem<GameNetworkSubclass> {
   public:
    PlayoutWorkElement(const PlayoutWorkElement&) = delete;
    PlayoutWorkElement& operator=(const PlayoutWorkElement&) = delete;

    PlayoutWorkElement(
        const PlayoutConfig<GameSubclass>& playout_config,
        PlayoutState& playout_state,
        PlayoutRunner& runner,
        NetworkDispatchQueue<GameNetworkSubclass>& work_queue) :
            playout_config_(playout_config),
            playout_state_(playout_state),
            runner_(runner),
            work_queue_(work_queue) {}

    virtual ~PlayoutWorkElement() override {};

    virtual void operator()(GameNetworkSubclass* local_network) const = 0;

   protected:
    const PlayoutConfig<GameSubclass>& playout_config_;
    PlayoutState& playout_state_;
    PlayoutRunner& runner_;
    NetworkDispatchQueue<GameNetworkSubclass>& work_queue_;
  };

  class FanoutWorkElement : public PlayoutWorkElement {
   public:
    FanoutWorkElement(const FanoutWorkElement&) = delete;
    FanoutWorkElement& operator=(const FanoutWorkElement&) = delete;

    FanoutWorkElement(
        const PlayoutConfig<GameSubclass>& playout_config,
        PlayoutState& playout_state, 
        PlayoutRunner& runner, 
        NetworkDispatchQueue<GameNetworkSubclass>& work_queue) : 
            PlayoutWorkElement(playout_config, playout_state, runner, 
                               work_queue) {}

    ~FanoutWorkElement() override {}

    void operator()(GameNetworkSubclass* local_network) const override {
      const GameSubclass& game(this->playout_state_.game);
      if (game.State() == games::GameState::kOver) {
        if (this->playout_state_.on_root) {
          LOG(FATAL) << "Cannot begin fanout from leaf.";
        }
        this->playout_state_.outcome = game.Outcome();
        return;
      }

      GameSubclass game_snapshot(game);
      int moves_n = game.CurrentMovesN();
      this->playout_state_.evals_remaining.store(moves_n,
                                                 std::memory_order_relaxed);
      this->playout_state_.eval_results.resize(moves_n);
      for (int i = 0; i < moves_n; ++i) {
        GameSubclass game_w_move(game_snapshot);
        game_w_move.MakeMove(i);

        this->work_queue_.AddWork(
            std::make_unique<EvaluateWorkElement>(
                this->playout_config_, this->playout_state_, this->runner_, 
                this->work_queue_, std::move(game_w_move), i));
      }
    }
  };

  // Runs the model and collects the outcome and current policy. These two 1D
  // vector values are concatenated together in the returned pointer. This also
  // returns the size of the policy vector (the outcome vector is known by
  // GameState::players_n()).
  static std::tuple<std::unique_ptr<float[]>, int> QueryModelForGameState(
      const GameSubclass& game, 
      GameNetworkSubclass& local_network,
      Cache& cache) {
    int policy_n = 0;
    if (game.State() == games::GameState::kOngoing) {
      policy_n = local_network.policy_output_sizes(game.PolicyClassI());
    }

    // We cache the current state's outcome + policy predictions by stacking
    // them together.
    int cached_values_n = GameSubclass::players_n() + policy_n;
    auto cached_output = std::make_unique<float[]>(cached_values_n);
    TempCacheKey temp_key(game.state_uid());
    if (!cache.TryLoad(temp_key, cached_output.get(), cached_values_n)) {
      local_network.SetConstants(local_network.input_constant_indices(),
                                 game.StateToMatrix());
      std::vector<nn::DynamicMatrix> model_outputs;
      if (policy_n > 0) {
        local_network.Outputs(
            {
                local_network.outcome_output_index(),
                local_network.policy_output_indices()[game.PolicyClassI()]
            },
            model_outputs);
        std::memcpy(
            cached_output.get() + GameSubclass::players_n(),
            model_outputs[1].data(),
            policy_n * sizeof(float));
      } else {
        local_network.Outputs(
            {
                local_network.outcome_output_index()
            },
            model_outputs);
      }

      std::memcpy(
          cached_output.get(),
          model_outputs[0].data(),
          GameSubclass::players_n() * sizeof(float));

      // If this fails, too bad.
      cache.TryStore(temp_key, cached_output.get(), cached_values_n);
    }
    return {std::move(cached_output), policy_n};
  }

  class EvaluateWorkElement : public PlayoutWorkElement {
   public:
    EvaluateWorkElement(const EvaluateWorkElement&) = delete;
    EvaluateWorkElement& operator=(const EvaluateWorkElement&) = delete;

    EvaluateWorkElement(
        const PlayoutConfig<GameSubclass>& playout_config,
        PlayoutState& playout_state,
        PlayoutRunner& runner,
        NetworkDispatchQueue<GameNetworkSubclass>& work_queue,
        GameSubclass&& game,
        int eval_index) :
            PlayoutWorkElement(playout_config, playout_state, runner, 
                               work_queue),
            game_(std::move(game)),
            eval_index_(eval_index) {}

    ~EvaluateWorkElement() override {}

    void operator()(GameNetworkSubclass* local_network) const override {
      if (game_.State() == games::GameState::kOngoing) {
        auto [model_output, policy_n] = PlayoutRunner::QueryModelForGameState(
            game_, *local_network, this->runner_.cache_);

        // Since we store the policy and outcome together with the outcome
        // first, we only care about the first GameSubclass::players_n() values
        // from the output, and from these we only care about the predicted odds
        // for the player who's turn it is..
        // 
        // Since the evaluation we just did was on state i + 1, not i, we have
        // to back out the odds for the player moving on i.
        int parent_player_i = ((this->playout_state_.game.CurrentPlayerI() 
            - game_.CurrentPlayerI()) + GameSubclass::players_n()) 
                % GameSubclass::players_n();
        this->playout_state_.eval_results[eval_index_] =
            {game_.state_uid(), model_output[parent_player_i]};
      } else {
        // If the game is over, lets just use the information known exactly
        // rather than ask the model.
        this->playout_state_.eval_results[eval_index_] =
            {game_.state_uid(), 
             game_.Outcome()[this->playout_state_.game.CurrentPlayerI()]};
      }

      uint32_t evals_remaining = this->playout_state_.evals_remaining.fetch_add(
          -1, std::memory_order_relaxed);
      if (evals_remaining > 1) return;

      this->work_queue_.AddWork(
          std::make_unique<SelectWorkElement>(
              this->playout_config_, this->playout_state_, this->runner_, 
              this->work_queue_));
    }

   private:
    const GameSubclass game_;
    const int eval_index_;
  };

  class SelectWorkElement : public PlayoutWorkElement {
   public:
    SelectWorkElement(const SelectWorkElement&) = delete;
    SelectWorkElement& operator=(const SelectWorkElement&) = delete;

    SelectWorkElement(
        const PlayoutConfig<GameSubclass>& playout_config,
        PlayoutState& playout_state,
        PlayoutRunner& runner,
        NetworkDispatchQueue<GameNetworkSubclass>& work_queue) :
            PlayoutWorkElement(playout_config, playout_state, runner, 
                               work_queue) {}

    ~SelectWorkElement() override {}

    void operator()(GameNetworkSubclass* local_network) const override {
      auto [model_output, policy_n] = PlayoutRunner::QueryModelForGameState(
          this->playout_state_.game, *local_network, this->runner_.cache_);

      // TODO: This calculation could also be cached for the whole iteration.
      //     Instead of caching the model outputs, we can cache this calculation
      //     once and Fanout can choose not to re-run it, just selecting the
      //     move directly. Obviously the noise-ing would happen every time, but
      //     conflating the outcome and policy terms doesn't have to.
      //     -
      //     Also add tail moves to be friendlier to the cache.

      int moves_n = this->playout_state_.eval_results.size();
      float uniform_prob = 1.0f / static_cast<float>(moves_n);
      std::vector<float> move_prop(moves_n, 0.0f);
      // Skip past the outcome part of the output and just fetch the policy.
      auto policy = std::span<float>(&(model_output[GameSubclass::players_n()]), 
                                     policy_n);

      // First step is to get a conflated distribution of predicted policy and
      // sub-game outcomes. We need to normalize these first.
      float policy_sum = 0.0f;
      float outcome_sum = 0.0f;
      for (int i = 0; i < moves_n; ++i) {
        move_prop[i] = this->playout_state_.game.PolicyForMoveI(policy, i);
        policy_sum += move_prop[i];
        outcome_sum += std::get<1>(this->playout_state_.eval_results[i]);
      }
      // Apply the temperature knobs to the policy and outcome for each move,
      // then conflate the discrete distributions: these are attempts at
      // measuring the same phenomenon in two different ways.
      for (int i = 0; i < moves_n; ++i) {
        float policy_f = 
            (move_prop[i] / policy_sum) 
                * (1 - this->playout_config_.policy_temp)
            + uniform_prob * this->playout_config_.policy_temp;
        float outcome_f = 
            (std::get<1>(this->playout_state_.eval_results[i]) / outcome_sum)
                * (1 - this->playout_config_.outcome_temp)
            + uniform_prob * this->playout_config_.outcome_temp;

        move_prop[i] = policy_f * outcome_f;
      }

      absl::BitGen gen;

      // One more pass to include noise in the final move proportions.
      AddNoise(gen, move_prop, this->playout_config_.noise_temp);

      // Sample from move_prop.
      int move_index = Sample(gen, move_prop);

      // Blocking TODO list:
      //     - Come up with a better way to include visit counts, and implement
      //       it.
      //     - Plumb all the temperature parameters through and such.
      //     - Test to make sure is actually learning.
      //     - Re-write to cache move_prop, not the model evaluations. This will
      //       work by FanoutWorkElement checking for the existence of a cached
      //       copy, and if it doesn't find it, creating an eval work element.
      //       Things proceed the same way except we don't cache model results,
      //       and select now caches and pushes the next fanout. Fanout and
      //       Select share a factored method that selects the next move for a 
      //       given playout.

      /*
      int best_move_i = -1;
      {
        // While this lock ensures that playouts revisiting the same state will
        // observe child state visit count sequentially, it does not ensure that
        // a playout at a different state that shares a child with this one
        // *will* observe visit counts sequentially. We assume this is rare
        // enough to be okay.
        auto this_state_uid = this->playout_state_.game.state_uid();
        auto lock = this->runner_.state_lock_.Lock(this_state_uid);
        int visit_count = this->runner_.visit_table_.Get(this_state_uid);

        float best_score = 0.0f;

        int move_i = 0;
        absl::BitGen bitgen;
        for (const auto& [state_uid, outcome_for_move] : 
             this->playout_state_.eval_results) {
          int visit_count_for_move = this->runner_.visit_table_.Get(state_uid);
          float visit_count_term = std::sqrtf(
              std::logf(static_cast<float>(this->playout_config_.n))
                  / static_cast<float>(visit_count_for_move + 1));
          
          
          // The score for the sub-game resulting from making move move_i.
          float score = 
              this->playout_config_.outcome_weight * outcome_for_move
              + this->playout_config_.policy_weight * policy_term
              + this->playout_config_.revisit_weight * visit_count_term;
          

          if (score >= best_score) {
            best_score = score;
            best_move_i = move_i;
          }

          ++move_i;
        }

        // We now know that best_move_i is the move we're going to make, so
        // we can increment the visit count to this sub-state before we actually
        // get there for the sake of other visitors to the current state.
        this->runner_.visit_table_.Inc(
            std::get<0>(this->playout_state_.eval_results[best_move_i]));
      }
      */

      // On the root move, record which option we took.
      if (this->playout_state_.on_root) {
        this->playout_state_.move_index = move_index;
        this->playout_state_.on_root = false;
      }

      this->playout_state_.game.MakeMove(move_index);

      this->work_queue_.AddWork(std::make_unique<FanoutWorkElement>(
          this->playout_config_, this->playout_state_, this->runner_,
          this->work_queue_));
    }

    static inline int Sample(absl::BitGen& bitgen, 
                             const std::vector<float>& prop) {
      int move_index = 0;
      
      float uniform = absl::Uniform(bitgen, 0.0f, 1.0f);
      float total = 0.0f;
      float prev_total = total;
      for (; move_index < prop.size(); ++move_index) {
        total += prop[move_index];
        if ((uniform >= prev_total) && (uniform < total)) break;
        prev_total = total;
      }
      // Handle round-off errors.
      return (move_index == prop.size())
          ? prop.size() - 1
          : move_index;
    }

    static inline void AddNoise(absl::BitGen& bitgen, 
                                std::vector<float>& move_prop, 
                                float noise_temp) {
      int moves_n = move_prop.size();
      std::vector<float> noise(moves_n, 0.0f);
      float noise_sum = 0.0f;
      float move_prop_sum = 0.0f;
      for (int i = 0; i < move_prop.size(); ++i) {
        noise[i] = absl::Uniform(bitgen, 0.0f, 1.0f);
        noise_sum += noise[i];
      }
      for (int i = 0; i < moves_n; ++i) {
        noise[i] = (noise[i] / noise_sum) * (1 - noise_temp)
            + (1.0f / static_cast<float>(moves_n)) * noise_temp;
        move_prop[i] *= noise[i];
        move_prop_sum += move_prop[i];
      }
      for (int i = 0; i < moves_n; ++i) {
        move_prop[i] /= move_prop_sum;
      }
    }
  };
};

}  // namespace internal
}  // namespace mcts
}  // namespace azah

#endif  // AZAH_MCTS_PLAYOUT_RUNNER_H_
