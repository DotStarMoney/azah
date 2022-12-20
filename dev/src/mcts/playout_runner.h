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
#include "lock_by_key.h"
#include "network_dispatch_queue.h"
#include "state_cache.h"
#include "visit_table.h"
#include "network_work_item.h"

namespace azah {
namespace mcts {

template <typename GameSubclass>
struct PlayoutConfig {
  // The game state to evaluate from.
  GameSubclass game;

  // Number of playouts to run.
  int n;

  // The linear weight applied to the outcome term.
  float outcome_weight;

  // The linear weight applied to the search policy term. 
  float policy_weight;

  // The linear weight applied to the revisit term.
  float revisit_weight;

  // Standard deviation of Gaussian noise added to the search policy term.
  float policy_noise;
};

template <typename GameSubclass>
struct PlayoutResult {
  // The outcome order-rotated s.t. the player who's move it is to make is in
  // the first row.
  nn::Matrix<GameSubclass::players_n(), 1> outcome;

  // The search proportion across predicted options actually made.
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
    for (auto& state_ptr : playout_states) {
      move_totals[state_ptr->move_index] += state_ptr->outcome[
          config.game.CurrentPlayerI()];

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

    // The number of times each move at the root state was visited across
    // playouts.
    std::vector<int> moves_searched(config.game.CurrentMovesN(), 0);
    for (const auto& state_ptr : playout_states) {
      ++(moves_searched[state_ptr->move_index]);
    }
    result.policy = config.game.MoveVisitCountToPolicy(moves_searched);
    
    for (int i = 0; i < config.game.CurrentMovesN(); ++i) {
      if (moves_searched[i] != 0) {
        move_totals[i] /= static_cast<float>(moves_searched[i]);
      } else {
        move_totals[i] = 0.0f;
      }
    }
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

    void operator()(GameNetworkSubclass* local_network) const {
      const GameSubclass& game(this->playout_state_.game);
      if (game.State() == games::GameState::kOver) {
        if (this->playout_state_.on_root) {
          LOG(FATAL) << "Cannot begin fanout from leaf.";
        }
        this->playout_state_.outcome = std::move(game.Outcome());
        return;
      }

      this->playout_state_.evals_remaining.store(game.CurrentMovesN(), 
                                                 std::memory_order_relaxed);
      this->playout_state_.eval_results.resize(game.CurrentMovesN());
      for (int i = 0; i < game.CurrentMovesN(); ++i) {
        GameSubclass game_w_move(game);
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
      policy_n = local_network.PolicyOutputSize(game.PolicyClassI());
    }

    // We cache the current state's outcome + policy predictions by stacking
    // them together.
    int cached_values_n = GameSubclass::players_n() + policy_n;
    auto cached_output = std::make_unique<float[]>(cached_values_n);
    TempCacheKey temp_key(game.state_uid());
    if (!cache.TryLoad(temp_key, cached_output.get(), cached_values_n)) {
      local_network.SetConstants(local_network.InputConstantIndices(),
                                 game.StateToMatrix());
      std::vector<nn::DynamicMatrix> model_outputs;
      if (policy_n > 0) {
        local_network.Outputs(
            {
                local_network.OutcomeOutputIndex(),
                local_network.PolicyOutputIndices()[game.PolicyClassI()]
            },
            model_outputs);
        std::memcpy(
            cached_output.get() + GameSubclass::players_n(),
            model_outputs[1].data(),
            policy_n * sizeof(float));
      } else {
        local_network.Outputs(
            {
                local_network.OutcomeOutputIndex()
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

    void operator()(GameNetworkSubclass* local_network) const {
      auto [model_output, policy_n] = PlayoutRunner::QueryModelForGameState(
          game_, *local_network, this->runner_.cache_);

      // Since we store the policy and outcome together with the outcome first,
      // we only care about the first GameSubclass::player_n() values from the
      // output, and from these we only care about the predicted odds for the
      // player who's turn it is.
      this->playout_state_.eval_results[eval_index_] = 
          {game_.state_uid(), 
           model_output[this->playout_state_.game.CurrentPlayerI()]};

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

    void operator()(GameNetworkSubclass* local_network) const {
      auto [model_output, policy_n] = PlayoutRunner::QueryModelForGameState(
          this->playout_state_.game, *local_network, this->runner_.cache_);
      // Skip past the outcome part of the output and just fetch the policy.
      auto policy = std::span<float>(&(model_output[GameSubclass::players_n()]), 
                                     policy_n);
      int best_move_i = -1;
      {
        // While this lock ensures that playouts revisiting the same state will
        // observe child state visit count sequentially, it does not ensure that
        // a playout at a different state that shares a child with this one
        // *will* observe visit counts sequentially. We assume this is rare
        // enough to be okay.
        auto lock = this->runner_.state_lock_.Lock(
            this->playout_state_.game.state_uid());

        float best_score = 0.0f;

        int move_i = 0;
        absl::BitGen bitgen;
        for (const auto& [state_uid, outcome_for_move] : 
             this->playout_state_.eval_results) {
          int visit_count_for_move = this->runner_.visit_table_.Get(state_uid);
          float visit_count_term = std::sqrtf(
              std::logf(static_cast<float>(this->playout_config_.n))
                  / static_cast<float>(visit_count_for_move + 1));
          
          float policy_for_move = this->playout_state_.game.PolicyForMoveI(
              policy, move_i);
          float policy_term = policy_for_move + absl::Gaussian(
              bitgen, 0.0f, this->playout_config_.policy_noise);
          if (policy_term < 0.0f) policy_term = 0.0f;

          // The score for the sub-game resulting from making move move_i.
          float score = 
              this->playout_config_.outcome_weight * outcome_for_move
              + this->playout_config_.policy_weight * policy_term
              + this->playout_config_.revisit_weight * visit_count_term;

          if (score > best_score) {
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

      // On the root move, record which option we took.
      if (this->playout_state_.on_root) {
        this->playout_state_.move_index = best_move_i;
        this->playout_state_.on_root = false;
      }

      this->playout_state_.game.MakeMove(best_move_i);

      this->work_queue_.AddWork(std::make_unique<FanoutWorkElement>(
          this->playout_config_, this->playout_state_, this->runner_,
          this->work_queue_));
    }
  };
};

}  // namespace mcts
}  // namespace azah

#endif  // AZAH_MCTS_PLAYOUT_RUNNER_H_
