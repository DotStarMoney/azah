#ifndef AZAH_MCTS_PLAYOUT_RUNNER_H_
#define AZAH_MCTS_PLAYOUT_RUNNER_H_

#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "../nn/data_types.h"
#include "../nn/init.h"
#include "Eigen/Core"
#include "lock_by_key.h"
#include "network_dispatch_queue.h"
#include "state_cache.h"
#include "visit_table.h"
#include "network_work_item.h"

// - Fill out SelectWorkElement
// - Add a work element for pure MCTS

namespace azah {
namespace mcts {

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

  void ClearModelCache() {
    cache_.Clear();
  }

  struct PlayoutConfig {
    // The game state to evaluate from.
    GameSubclass game;

    // Number of playouts to run.
    int n;

    // If true, don't query the model and just run vanilla MCTS.
    bool mcts_only;

    // The linear weight applied to the search policy term. 
    float policy_weight;
  
    // The linear weight applied to the revisit term.
    float revisit_weight;
  };

  struct PlayoutResult {
    // The outcome order-rotated s.t. the player who's move it is to make is in
    // the first row.
    nn::Matrix<GameSubclass::players_n(), 1> outcome;

    // The search proportion across predicted options actually made.
    nn::DynamicMatrix policy;

    // The move option with the greatest average win proportion across playouts.
    int max_option_i;
  };

  PlayoutResult Playout(const PlayoutConfig& config,
                        NetworkDispatchQueue<GameNetworkSubclass>& work_queue) {
    work_queue.Drain();
    std::vector<PlayoutState> playout_states(config.n, 
                                             PlayoutState(config.game));
    for (auto& state : playout_states) {
      work_queue.AddWork(std::make_unique<FanoutWorkElement>(state, *this, 
                                                             work_queue));
    }
    work_queue.Drain();
    visit_table_.Clear();

    PlayoutResult result;

    result.outcome = nn::init::Zeros<GameSubclass::players_n(), 1>();
    // The accumulated chance that the player who's turn it currently is wins
    // after making one of the possible moves. We'll later divide this by the
    // number of visits to each move and select the highest one.
    std::vector<float> move_totals(0, config.game.CurrentMovesN());
    for (auto& state : playout_states) {
      // Rotate the outcome array to move the player who's turn it currently is
      // into the first position.
      std::rotate(
          state.outcome, 
          state.outcome + config.game.CurrentPlayerI(), 
          state.outcome + GameSubclass::players_n());

      move_totals[state.move_index] += state.outcome[0];

      result.outcome = (result.outcome.array()
          + Eigen::Map<nn::Matrix<GameSubclass::players_n(), 1>>(state.outcome))
              .matrix();
    }
    result.outcome /= static_cast<float>(config.n);
    
    // The number of times each move at the root state was visited across
    // playouts.
    std::vector<int> moves_searched(0, config.game.CurrentMovesN());
    for (const auto& state : playout_states) {
      ++(moves_searched[state.move_index]);
    }
    result.policy = config.game.MoveVisitCountToPolicy(moves_searched);
    
    for (int i = 0; i < config.game.CurrentMovesN(); ++i) {
      move_totals[i] /= static_cast<float>(moves_searched[i]);
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
    PlayoutState(const GameSubclass& game) : 
        move_index(-1), evals_remaining(-1), on_root(true), game(game) {}

    // The playout's outcome.
    std::array<float, GameSubclass::players_n()> outcome;

    // Index of the move made after fanning out the root for this playout.
    int move_index;

    // Whether or not the playout is currently evaluating the root.
    bool on_root;

    // The state of the game currently being evaluated in this playout.
    GameSubclass game;

    // The results of evaluating each move in the current game state.
    std::vector<float> eval_results;

    // Number of completed evaluations in the current fanout for this playout.
    std::atomic<uint32_t> evals_remaining;
  };

  class PlayoutWorkElement : public GameNetworkWorkItem<GameNetworkSubclass> {
   public:
    PlayoutWorkElement(const PlayoutWorkElement&) = delete;
    PlayoutWorkElement& operator=(const PlayoutWorkElement&) = delete;

    PlayoutWorkElement(
        PlayoutState& playout_state,
        PlayoutRunner& runner,
        NetworkDispatchQueue<GameNetworkSubclass>& work_queue) :
            playout_state_(playout_state),
            runner_(runner),
            work_queue_(work_queue) {}

    virtual void operator()(GameNetworkSubclass* local_network) const = 0;

   protected:
    PlayoutState& playout_state_;
    PlayoutRunner& runner_;
    NetworkDispatchQueue<GameNetworkSubclass>& work_queue_;
  };

  class FanoutWorkElement : public PlayoutWorkElement {
   public:
     FanoutWorkElement(const FanoutWorkElement&) = delete;
     FanoutWorkElement& operator=(const FanoutWorkElement&) = delete;

     FanoutWorkElement(
        PlayoutState& playout_state, 
        PlayoutRunner& runner, 
        NetworkDispatchQueue<GameNetworkSubclass>& work_queue) : 
            PlayoutWorkElement(playout_state, runner, work_queue) {}

    void operator()(GameNetworkSubclass* local_network) const {
      const GameSubclass& game(this->playout_state_.game);
      if (game.State() == GameSubclass::GameState::kOver) {
        if (this->playout_state_.on_root) {
          LOG(FATAL) << "Cannot begin fanout from leaf.";
        }
        this->playout_state_.outcome = std::move(game.Outcome());
        return;
      }

      this->playout_state_.evals_remaining.store(0, std::memory_order_relaxed);
      this->playout_state_.eval_results.resize(game.CurrentMovesN());
      for (int i = 0; i < game.CurrentMovesN(); ++i) {
        GameSubclass game_w_move(game);
        game_w_move.MakeMove(i);

        this->work_queue_.AddWork(
            std::make_unique<EvaluateWorkElement>(
                this->playout_state_, this->runner_, this->work_queue_,
                std::move(game_w_move), i));
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
    std::vector<nn::DynamicMatrixRef> outputs;
    if (game.State() == GameSubclass::kOngoing) {
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
      local_network.Outputs(
          {
              local_network.OutcomeOutputIndex(),
              local_network.PolicyOutputIndices()[game.PolicyClassI()]
          },
          model_outputs);

      std::memcpy(
          cached_output.get(),
          model_outputs[0].data(),
          GameSubclass::players_n());

      if (policy_n != 0) {
        std::memcpy(
            cached_output.get() + GameSubclass::players_n(),
            model_outputs[1].data(),
            policy_n);
      }

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
        PlayoutState& playout_state,
        PlayoutRunner& runner,
        NetworkDispatchQueue<GameNetworkSubclass>& work_queue,
        GameSubclass&& game,
        int eval_index) :
            PlayoutWorkElement(playout_state, runner, work_queue),
            game_(std::move(game)),
            eval_index_(eval_index) {}

    void operator()(GameNetworkSubclass* local_network) const {
      auto [model_output, policy_n] = PlayoutRunner::QueryModelForGameState(
          game_, *local_network, this->runner_.cache_);

      // Since we store the policy and outcome together with the outcome first,
      // we only care about the first GameSubclass::player_n() values from the
      // output, and from these we only care about the predicted odds for the
      // player who's turn it is.
      this->playout_state_.eval_results_[eval_index_] = 
          model_output[game_.CurrentPlayerI()];

      uint32_t evals_remaining = this->playout_state_.evals_remaining.fetch_add(
          -1, std::memory_order_relaxed);
      if (evals_remaining > 0) return;

      this->work_queue_.AddWork(
          std::make_unique<SelectWorkElement>(
              this->playout_state_, this->runner_, this->work_queue_));
    }
   
   private:
    const GameSubclass game_;
    int eval_index_;
  };

  // SelectWorkElement
  //   - Gather the visit counts for each move tried in a locking fashion, use 
  //     the table to select the next move. Make that move, and pass /
  //     push FanoutWorkElement(non-root). If root, increment atomic moves
  //     counter.

  class SelectWorkElement : public PlayoutWorkElement {
   public:
    SelectWorkElement(const SelectWorkElement&) = delete;
    SelectWorkElement& operator=(const SelectWorkElement&) = delete;

    SelectWorkElement(
        PlayoutState& playout_state,
        PlayoutRunner& runner,
        NetworkDispatchQueue<GameNetworkSubclass>& work_queue) :
            PlayoutWorkElement(playout_state, runner, work_queue) {}

    void operator()(GameNetworkSubclass* local_network) const {
      //
      //
      //
    }
  };
};

}  // namespace mcts
}  // namespace azah

#endif  // AZAH_MCTS_PLAYOUT_RUNNER_H_
