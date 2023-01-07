#ifndef AZAH_MCTS_RL_PLAYER_H_
#define AZAH_MCTS_RL_PLAYER_H_

#include <stddef.h>

#include <memory>
#include <ostream>
#include <vector>

#include "../games/game.h"
#include "../games/game_network.h"
#include "../nn/adam.h"
#include "../nn/data_types.h"
#include "absl/container/flat_hash_map.h"
#include "glog/logging.h"
#include "self_play.h"
#include "work_queue.h"

namespace azah {
namespace mcts {

template <games::AnyGameType Game, games::GameNetworkType GameNetwork>
class RLPlayer {
 public:
  struct Options {
    static constexpr std::size_t kDefaultAsyncDispatchQueueLength = 256;

    std::size_t async_dispatch_queue_length = kDefaultAsyncDispatchQueueLength;
  };

  RLPlayer(std::size_t replicas_n, const Options& options = Options()) : 
      work_queue_(replicas_n, options.async_dispatch_queue_length) {
    ResetInternal(replicas_n);
  }

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

    // A multiplier on the upper-confidence-bound that encourages exploration
    // when higher.
    float exploration_scale;
  };

  struct EvaluateResult {
    // The belief in the outcome of the game. 
    std::array<float, Game::players_n()> predicted_outcome;

    // The belief of the move that should be made for the current player to
    // maximize their odds of winning.
    std::vector<float> predicted_move;
  };

  EvaluateResult Evaluate(const Game& position, 
                          const SelfPlayOptions& self_play_options) {
    if (position.State() == games::GameState::kOver) {
      LOG(FATAL) << "Game is over.";
    }
    auto self_play_config = SelfPlayOptionsToConfig(false, self_play_options);
    auto replica_moves = CollectReplicaMoves(position, self_play_config);

    // Average the outcomes and search policies into the first replica.
    for (std::size_t i = 1; i < replicas_.size(); ++i) {
      replica_moves[0][0].outcome += replica_moves[i][0].outcome;
      replica_moves[0][0].search_policy += replica_moves[i][0].search_policy;
    }
    replica_moves[0][0].outcome /= static_cast<float>(replicas_.size());
    replica_moves[0][0].search_policy /= static_cast<float>(replicas_.size());

    // Copy the results out.
    EvaluateResult result;
    for (std::size_t i = 0; i < position.CurrentMovesN(); ++i) {
      result.predicted_move.push_back(
          position.PolicyForMoveI(replica_moves[0][0].search_policy, i));
    }
    for (std::size_t i = 0; i < Game::players_n(); ++i) {
      result.predicted_outcome[i] = replica_moves[0][0].outcome(i, 0);
    }
    return result;
  }

  struct TrainResult {
    // Average softmax cross entropy across replicas + moves between true search
    // policies and predicted search policies.
    float policy_loss;

    // Average softmax cross entropy across replicas + moves between true
    // outcomes and predicted outcomes.
    float outcome_loss;

    friend std::ostream& operator<<(std::ostream& os, const TrainResult& x) {
      os << "{outcome_loss=" << x.outcome_loss << ",policy_loss="
          << x.policy_loss << "}";
      return os;
    }
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
        .one_hot_breakover_moves_n = 
            self_play_options.one_hot_breakover_moves_n,
        .exploration_scale = self_play_options.exploration_scale};
  }

  class ReplicaSelfPlayerFn : public internal::WorkQueueElement {
   public:
    ReplicaSelfPlayerFn(const Game& position, const self_play::Config& config,
                        GameNetwork* network,
                        std::vector<self_play::MoveOutcome<Game>>* moves) :
        position_(position), config_(config), network_(network), 
        moves_(moves) {}

    void run() override {
      *moves_ = std::move(self_play::SelfPlay(config_, position_, network_));
    }

   private:
    const Game& position_;
    const self_play::Config& config_;
    GameNetwork* network_;
    std::vector<self_play::MoveOutcome<Game>>* moves_;
  };

  class ReplicaSGDFn : public internal::WorkQueueElement {
   public:
    ReplicaSGDFn(
        float learning_rate,
        Replica& replica,
        const std::vector<const self_play::MoveOutcome<Game>*>& all_moves,
        TrainResult& replica_loss) :
            learning_rate_(learning_rate), replica_(replica),
            all_moves_(all_moves), replica_loss_(replica_loss) {}

    void run() override {
      GameNetwork& network = replica_.network;
      
      // These will be re-used between updates.
      std::vector<nn::DynamicMatrix> var_grad;
      std::vector<uint32_t> var_grad_i;
      std::vector<float> losses;

      // Since not all variables are guaranteed to have gradients for each row
      // in updates, we have to do some extra bookkeeping to sum the terms of
      // each variable's average gradients.
      absl::flat_hash_map<uint32_t, int> var_index_to_vec_index;

      // Parallel arrays
      std::vector<nn::DynamicMatrix> acc_grads;
      std::vector<uint32_t> acc_index;
      std::vector<int> acc_grad_count;

      uint32_t outcome_target_index = network.outcome_target_constant_index();
      uint32_t outcome_loss_index = network.outcome_loss_target_index();
      for (auto update : all_moves_) {
        // Step 1: Calculate gradients for the given update.
        uint32_t policy_target_index = 
            network.policy_target_constant_indices()[
                update->search_policy_class_i];

        network.SetConstants(network.input_constant_indices(), 
                             update->state_inputs);
        network.SetConstants({policy_target_index}, {update->search_policy});
        network.SetConstants({outcome_target_index}, {update->outcome});

        uint32_t policy_loss_index =
            network.policy_loss_target_indices()[update->search_policy_class_i];

        network.Gradients({policy_loss_index, outcome_loss_index}, var_grad_i, 
                          var_grad, losses);
        
        // Step 2: Accumulate the gradients into the averages.
        for (int i = 0; i < var_grad.size(); ++i) {
          uint32_t grad_index = var_grad_i[i];
          nn::DynamicMatrix& grad = var_grad[i];
          auto [iter, is_new] = var_index_to_vec_index.insert(
              {grad_index, acc_grads.size()});
          if (is_new) {
            acc_grads.push_back(std::move(grad));
            acc_index.push_back(grad_index);
            acc_grad_count.push_back(1);
          } else {
            acc_grads[iter->second] += grad;
            (acc_grad_count[iter->second])++;
          }
        }
        replica_loss_.policy_loss += losses[0];
        replica_loss_.outcome_loss += losses[1];
      }

      for (std::size_t i = 0; i < acc_grads.size(); ++i) {
        acc_grads[i] /= static_cast<float>(acc_grad_count[i]);
      }
      replica_loss_.policy_loss /= static_cast<float>(all_moves_.size());
      replica_loss_.outcome_loss /= static_cast<float>(all_moves_.size());

      replica_.opt.Update(learning_rate_, acc_index, acc_grads, 
                          replica_.network);
    }

   private:
    const float learning_rate_;
    Replica& replica_;
    const std::vector<const self_play::MoveOutcome<Game>*>& all_moves_;
    TrainResult& replica_loss_;
  };

  inline std::vector<std::vector<self_play::MoveOutcome<Game>>> 
      CollectReplicaMoves(const Game& position, 
                          const self_play::Config& self_play_config) {
    std::vector<std::vector<self_play::MoveOutcome<Game>>> replica_moves(
        replicas_.size());
    std::vector<self_play::MoveOutcome<Game>>* moves_ptr = &(replica_moves[0]);
    for (auto& replica : replicas_) {
      work_queue_.AddWork(std::make_unique<ReplicaSelfPlayerFn>(
          position, self_play_config, &(replica->network), moves_ptr++));
    }
    work_queue_.Drain();
    return replica_moves;
  }

  TrainResult TrainIteration(
      float learning_rate, const self_play::Config& self_play_config) {
    // This is effectively a nested list of training examples and needs to
    // outlive gradient accumulation for obvious reasons.
    auto replica_moves = CollectReplicaMoves(root_game_, self_play_config);
    
    // Flatten the list of move outcomes.
    std::vector<const self_play::MoveOutcome<Game>*> all_moves;
    for (std::size_t rep_i = 0; rep_i < replicas_.size(); ++rep_i) {
      for (auto& move : replica_moves[rep_i]) {
        all_moves.push_back(&move);
      }
    }
    
    // Update the replicas.
    std::vector<TrainResult> replica_losses(replicas_.size(), {0.0f, 0.0f});
    for (std::size_t i = 0; i < replicas_.size(); ++i) {
      work_queue_.AddWork(std::make_unique<ReplicaSGDFn>(
          learning_rate, *(replicas_[i]), all_moves, replica_losses[i]));
    }
    work_queue_.Drain();

    // Average the loss over the replicas.
    TrainResult final_loss{0.0f, 0.0f};
    for (std::size_t i = 0; i < replicas_.size(); ++i) {
      final_loss.policy_loss += replica_losses[i].policy_loss;
      final_loss.outcome_loss += replica_losses[i].outcome_loss;
    }
    final_loss.policy_loss /= static_cast<float>(replicas_.size());
    final_loss.outcome_loss /= static_cast<float>(replicas_.size());

    return final_loss;
  }
};

template <games::AnyGameType Game, games::GameNetworkType GameNetwork>
const Game RLPlayer<Game, GameNetwork>::root_game_ = Game();

}  // namespace mcts
}  // namespace azah

#endif  // AZAH_MCTS_RL_PLAYER_H_
