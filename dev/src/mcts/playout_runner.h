#ifndef AZAH_MCTS_PLAYOUT_RUNNER_H_
#define AZAH_MCTS_PLAYOUT_RUNNER_H_

#include <algorithm>
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

//
// Need to add a better way to reference certain parts of a model from Game
// subclasses
//
// Then have to finish running the model in the evaluate fn and storing it.

namespace azah {
namespace mcts {

template <typename GameSubclass, typename GameNetworkSubclass, int Shards, 
          int CacheBlocks, int CacheRowsPerBlock, int CacheValuesPerRow>
class PlayoutRunner {
  using Cache = StateCache<std::string, float, CacheBlocks, CacheRowsPerBlock,
                           CacheValuesPerRow>;
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
    // The accumulated chance that the player who's move it currently is wins
    // after making one of the possible moves. We'll later divide this by the
    // number of visits to each move and select the highest one.
    std::vector<float> move_totals(0, config.game.CurrentMovesN());
    for (auto& state : playout_states) {
      // Rotate the outcome array to move the player who's move it currently is
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
        move_index(-1), fanout_count(-1), on_root(true), game(game) {}

    // The playout's outcome.
    float outcome[GameSubclass::players_n()];

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

  // EvalWorkElement
  //   - Run the model with the game vector. Deposit the est. win outcome and
  //     visit count into the table. If we've deposited all outcomes, push
  //     SelectWorkElement.
  //
  // SelectWorkElement
  //   - Gather the visit counts for each move tried in a locking fashion, use 
  //     the table to select the next move. Make that move, and pass /
  //     push FanoutWorkElement(non-root). If root, increment atomic moves
  //     counter.


  class PlayoutWorkElement : public NetworkWorkItem<GameNetworkSubclass> {
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

  static constexpr int kOutcomeNetworkOutput = 0;

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
            game_(std::move(game),
            eval_index_(eval_index)) {}

    void operator()(GameNetworkSubclass* local_network) const {
      int policy_n = 0;
      std::vector<nn::DynamicMatrixRef> outputs;
      if (game_.State() == GameSubclass::kOngoing) {
        //local_network->GetVariables({game_.PolicyClassI()}, outputs);
        //policy_n = outputs[0].size();
 
        //
        // Network needs way to return size of outputs. Get this for policy_n
        //

        //
        // Need better ways to indicate specific inputs and outputs. Should add
        // a few methods to game which allow you to index specific inputs / 
        // targets / policy output / outcome output / etc... 
        //
      }

      Cache::TempKey temp_key(game_.state_uid());
      int cached_values_n = 1 + policy_n;
      auto cached_output = std::make_unique<float[]>(cached_values_n);
      if (this->runner_.state_lock_.TryLoad(temp_key, &(cached_output[0]),
                                            cached_values_n)) {
        this->playout_state_.eval_results_[eval_index_] = cached_output[0];
      } else {
        auto inputs = game_.StateToMatrix();
        std::vector<nn::DynamicMatrixRef> model_inputs;
        local_network->GetConstantsByTag(game_.inputs_model_tag(), 
                                         model_inputs);
        if (inputs.size() != model_inputs.size()) {
          LOG(FATAL) << "Count of network constants with inputs_model_tag() "
                     << "does not match that returned by game StateToMatrix()";
        }
        for (int i = 0; i < model_inputs.size(); ++i) {
          model_inputs[i] = inputs[i];
        }
        std::vector<nn::DynamicMatrix> outcome;
        local_network->Output({kOutcomeNetworkOutput}, outcome);



      }

      // Use current game policy class to stash policy as well.



      uint32_t evals_remaining = this->playout_state_.evals_remaining.fetch_add(
          -1, std::memory_order_relaxed);
      if (evals_remaining > 0) return;

      // Add select work item

    }
   
   private:
    const GameSubclass game_;
    int eval_index_;
  };
};

}  // namespace mcts
}  // namespace azah

#endif  // AZAH_MCTS_PLAYOUT_RUNNER_H_
