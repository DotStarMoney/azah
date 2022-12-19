#ifndef AZAH_MCTS_PLAYOUT_RUNNER_H_
#define AZAH_MCTS_PLAYOUT_RUNNER_H_

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "../nn/data_types.h"
#include "../nn/init.h"
#include "Eigen/Core"
#include "lock_by_key.h"
#include "network_dispatch_queue.h"
#include "state_cache.h"
#include "visit_table.h"
#include "network_work_item.h"

namespace azah {
namespace mcts {

template <typename GameSubclass, typename GameNetworkSubclass, int Shards, 
          int CacheBlocks, int CacheRowsPerBlock, int CacheValuesPerRow>
class PlayoutRunner {
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
    nn::Matrix<GameSubclass::players_n(), 1> outcome;
    nn::DynamicMatrix policy;
  };

  PlayoutResult Playout(const PlayoutConfig& config,
                        NetworkDispatchQueue<GameNetworkSubclass>& work_queue) {
    work_queue.Drain();
    std::vector<PlayoutState> playout_states(config.n, 
                                             PlayoutState(config.game));
    for (auto& state : playout_states) {
      work_queue.AddWork(std::make_unique<FanoutWorkElement>(state));
    }
    work_queue.Drain();
    visit_table_.Clear();

    PlayoutResult result;

    result.outcome = nn::init::Zeros<GameSubclass::players_n(), 1>();
    for (const auto& state : playout_states) {
      result.outcome = (result.outcome.array()
          + Eigen::Map<nn::Matrix<GameSubclass::players_n(), 1>>(
                state.outcome.data())).matrix();
    }
    result.outcome /= static_cast<float>(config.n);

    std::vector<int> moves_searched(0, config.game.CurrentMovesN());
    for (const auto& state : playout_states) {
      ++(moves_searched[state.move_index])
    }
    result.policy = config.game.MoveVisitCountToPolicy(moves_searched);
    
    return result;
  }

 private:
  StateCache<std::string, float, CacheBlocks, CacheRowsPerBlock, 
             CacheValuesPerRow> cache_;
  VisitTable<std::string, Shards> visit_table_;
  LockByKey<std::string, Shards> state_lock_;

  struct PlayoutState {
    PlayoutState(const GameSubclass& game) : 
        move_index(-1), fanout_count(-1), on_root(true), game(game) {}

    // The playout's outcome.
    array<float, GameSubclass::players_n()> outcome;

    // Index of the move made after fanning out the root for this playout.
    int move_index;

    // Number of completed evaluations in the current fanout for this playout.
    std::atomic<uint32_t> fanout_count;

    // Whether the playout is currently evaluating the root.
    bool on_root;

    // The state of the game currently being evaluated in this playout.
    GameSubclass game;
  };
  
  // FanoutWorkElement
  //   - Checks to see if over, if so:
  //     + If its the root, fail.
  //     + If its not the root, accumulate the win result + source index into
  //       the table, ret.
  //   - else:
  //     + For all possible moves:
  //       + push EvalWorkElement. Each should have a copy of an atomic
  //         counter, and a vector + index to deposit their outcome.
  //
  // EvalWorkElement
  //   - Run the model with the game vector. Deposit the win outcome into the
  //     table, and the visit count. If we've deposited all outcomes, push
  //     SelectWorkElement.
  //
  // SelectWorkElement
  //   - Use the table to select the next move. Make that move, and pass /
  //     push FanoutWorkElement(non-root). If root, increment atomic moves
  //     counter.

  class FanoutWorkElement : public NetworkWorkItem {
   public:
    FanoutWorkElement(const FanoutWorkElement&) = delete;
    FanoutWorkElement& operator=(const FanoutWorkElement&) = delete;

    FanoutWorkElement(PlayoutState& playout_state) : 
       playout_state_(playout_state) {
      
    }

   private:
    PlayoutState& playout_state_;
  };
};

}  // namespace mcts
}  // namespace azah

#endif  // AZAH_MCTS_PLAYOUT_RUNNER_H_
