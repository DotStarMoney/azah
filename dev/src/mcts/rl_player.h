#ifndef AZAH_MCTS_RL_PLAYER_H_
#define AZAH_MCTS_RL_PLAYER_H_

#include <stddef.h>

#include <memory>
#include <ostream>
#include <vector>

#include "../nn/adam.h"
#include "self_play.h"
#include "work_queue.h"

namespace azah {
namespace mcts {

template <typename Game, typename GameNetwork>
class RLPlayer {
 public:
  struct Options {
    static constexpr std::size_t kDefaultAsyncDispatchQueueLength = 65536;
    static constexpr std::size_t kDefaultThreadsN = 4;

    std::size_t threads_n = kDefaultThreadsN;
    std::size_t async_dispatch_queue_length = kDefaultAsyncDispatchQueueLength;
  };

  RLPlayer(std::size_t replicas_n, const Options& options = Options()) : 
      work_queue_(options.threads_n, options.async_dispatch_queue_length) {
    ResetInternal(replicas_n);
  }

  struct TrainResult {
    // Average softmax cross entropy across replicas + moves between true search
    // policies and predicted search policies.
    float policy_loss;

    // Average softmax cross entropy across replicas + moves between true outcomes
    // and predicted outcomes.
    float outcome_loss;

    friend std::ostream& operator<<(std::ostream& os, const TrainResult& x) {
      os << "{outcome_Loss=" << x.outcome_loss << ",policy_loss="
          << x.policy_loss << "}";
      return os;
    }
  };

  struct SelfPlayOptions {
    // The SGD learning rate used when training.
    //
    // AlphaZero starts this at 0.2, and drops it by x0.1 multiple times.
    float learning_rate;

    // The total number of MCTS simulations to perform per-move.
    //
    // AlphaZero uses 800.
    int simulations_n;

    // The alpha value of Dirichlet noise added to the root search policy. Noise
    // is generated at every visit to the root.
    //
    // AlphaZero uses 0.3 for chess.
    float root_noise_alpha;

    // The amount interpolated between the predicted root policy and root noise.
    //
    // AlphaZero uses 0.25.
    float root_noise_lerp;

    // One-hot policy breakover threshold: the number of moves after which the
    // search policies returned from self-play are just one-hot of the maximum
    // valued move.
    //
    // AlphaZero uses 30.
    int one_hot_breakover_moves_n;

    // A multiplier on the upper-confidence-bound that encourages exploration when
    // higher.
    float exploration_scale;
  };

  TrainResult Train(int games_n, const SelfPlayOptions& self_play_options) {
    auto self_play_config = SelfPlayOptionsToConfig(true, self_play_options);

    TrainResult losses{0.0f, 0.0f};
    for (int i = 0; i < games_n; ++i) {
      auto iter_losses = TrainIteration(self_play_options.learning_rate,
                                        self_play_config);
      losses.policy_loss += iter_losses.policy_loss;
      losses.outcome_loss += iter_losses.outcome_loss;
    }

    losses.policy_loss /= static_cast<float>(games_n);
    losses.outcome_loss /= static_cast<float>(games_n);
    return losses;
  }

  void Reset() {
    ResetInternal(replicas_.size());
  }

 private:
  internal::WorkQueue work_queue_;
  static const Game root_game_;

  struct Replica {
    Replica() : opt(network) {}
    GameNetwork network;
    nn::Adam opt;
  };
  std::vector<std::unique_ptr<Replica>> replicas_;

  void ResetInternal(std::size_t n) {
    replicas_.clear();
    for (std::size_t i = 0; i < n; ++i) {
      replicas_.push_back(std::make_unique<Replica>());
    }
  }

  static inline self_play::Config SelfPlayOptionsToConfig(
      bool full_play, const SelfPlayOptions& self_play_options) {
    return {
        .simulations_n = self_play_options.simulations_n,
        .full_play = full_play,
        .root_noise_alpha = self_play_options.root_noise_alpha,
        .root_noise_lerp = self_play_options.root_noise_lerp,
        .one_hot_breakover_moves_n = self_play_options.one_hot_breakover_moves_n,
        .exploration_scale = self_play_options.exploration_scale};
  }

  class ReplicaSelfPlayerFn : public internal::WorkQueueElement {
   public:
    ReplicaSelfPlayerFn(const self_play::Config& config,
                        GameNetwork* network,
                        std::vector<self_play::MoveOutcome<Game>>* moves) :
        config_(config), network_(network), moves_(moves) {}

    void run() override {
      *moves_ = std::move(self_play::SelfPlay(
          config_, RLPlayer<Game, GameNetwork>::root_game_, network_));
    }

   private:
    const self_play::Config& config_;
    GameNetwork* network_;
    std::vector<self_play::MoveOutcome<Game>>* moves_;
  };

  TrainResult TrainIteration(
      float learning_rate, const self_play::Config& self_play_config) {
    // This is effectively a nested list of training examples and needs to outlive
    // gradient accumulation for obvious reasons.
    std::vector<std::vector<self_play::MoveOutcome<Game>>> replica_moves(
        replicas_.size(), {});

    {
      std::vector<self_play::MoveOutcome<Game>>* moves_ptr = &(replica_moves[0]);
      for (auto& replica : replicas_) {
        work_queue_.AddWork(std::make_unique<ReplicaSelfPlayerFn>(
            self_play_config, &(replica.network), moves_ptr++));
      }
      work_queue_.Drain();
    }
    
    std::vector<self_play::MoveOutcome<Game>*> all_moves;
    for (std::size_t rep_i; rep_i < replicas_.size(); ++rep_i) {
      for (auto& move : replica_moves[rep_i]) {
        all_moves.push_back(&move);
      }
    }
    
    //
    // Update replicas with all_moves, return losses
    //

    //
    // return losses
    //
  }


};

template <typename Game, typename GameNetwork>
const Game RLPlayer<Game, GameNetwork>::root_game_ = Game();

}  // namespace mcts
}  // namespace azah

#endif  // AZAH_MCTS_RL_PLAYER_H_
