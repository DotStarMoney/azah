#ifndef AZAH_MCTS_SELF_PLAYER_H_
#define AZAH_MCTS_SELF_PLAYER_H_

#include <stdint.h>

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

#include "../nn/adam.h"
#include "../nn/data_types.h"
#include "absl/container/flat_hash_map.h"
#include "network_dispatch_queue.h"
#include "game_network_work_item.h"
#include "glog/logging.h"
#include "playout_runner.h"

namespace azah {
namespace mcts {

template <typename GameSubclass>
struct EvaluateResult {
  const std::array<float, GameSubclass::players_n()> projected_outcome;
  const int best_move_option_i;
};

struct SelfPlayConfig {
  // Number of playouts to run.
  int playouts_n;

  // Learning rate used in SGD.
  float learning_rate;

  // The linear weight applied to the outcome term.
  float outcome_weight;

  // The linear weight applied to the search policy term. 
  float policy_weight;

  // The linear weight applied to the revisit term.
  float revisit_weight;

  // Standard deviation of Gaussian noise added to the search policy term.
  float policy_noise;
};

template <typename GameSubclass, typename GameNetworkSubclass, int Shards,
          int CacheBlocks, int CacheRowsPerBlock, int DispatchQueueLength>
class SelfPlayer {
 private:
  using DispatchQueue = internal::NetworkDispatchQueue<GameNetworkSubclass>;
  using PlayoutRunner = internal::PlayoutRunner<
      GameSubclass, GameNetworkSubclass, Shards, CacheBlocks, 
      CacheRowsPerBlock>;

 public:
  SelfPlayer(int threads) :
      primary_network_(new GameNetworkSubclass()),
      sgd_(new nn::Adam(*primary_network_)),
      work_queue_(new DispatchQueue(threads, DispatchQueueLength)), 
      playout_runner_(new PlayoutRunner()) {
    set_network_variables(*primary_network_);
  }

  SelfPlayer(
      int threads, 
      const GameNetworkSubclass& init_network) :
          primary_network_(new GameNetworkSubclass()),
          sgd_(new nn::Adam(*primary_network_)),
          work_queue_(new DispatchQueue(threads, DispatchQueueLength)),
          playout_runner_(new PlayoutRunner()) {
    set_network_variables(init_network);
  }

  void set_network_variables(const GameNetworkSubclass& network) {
    std::vector<nn::ConstDynamicMatrixRef> variables;
    network.GetVariables({}, variables);
    primary_network_->SetVariables({}, variables);
    work_queue_->SetAllVariables(variables);
  }

  const GameNetworkSubclass& get_network() const {
    return *primary_network_;
  }

  EvaluateResult<GameSubclass> EvaluatePosition(const GameSubclass& game, 
                                                const SelfPlayConfig& config) {
    if (game.State() == games::GameState::kOver) {
      LOG(FATAL) << "Game is over.";
    }
    auto playout_config = SelfPlayConfigToPlayoutConfig(game, config);

    internal::PlayoutResult<GameSubclass> result = playout_runner_->Playout(
        playout_config, *work_queue_);

    std::array<float, GameSubclass::players_n()> outcome;
    // We rotate the outcome so that it no longer starts with the current
    // player's move.
    for (int i = 0; i < GameSubclass::players_n(); ++i) {
      outcome[i] = result.outcome(
          (i - game.CurrentPlayerI() + GameSubclass::players_n())
              % GameSubclass::players_n(),
          0);
    }

    return {.projected_outcome = std::move(outcome), 
            .best_move_option_i = result.max_option_i};
  }

  struct TrainResult {
    // Average softmax cross entropy across games between true search policies
    // and predicted search policies.
    float policy_loss;
    
    // Average softmax cross entropy across games between true outcomes and
    // predicted outcomes.
    float outcome_loss;
  };

  void Train(int games_n, const SelfPlayConfig& config) {
    auto playout_config = SelfPlayConfigToPlayoutConfig(GameSubclass(), config);
    TrainResult losses{0.0f, 0.0f};
    for (int i = 0; i < games_n; ++i) {
      auto iter_losses = TrainIteration(playout_config, config.learning_rate);
      losses.policy_loss += iter_losses.policy_loss;
      losses.outcome_loss += iter_losses.outcome_loss;
    }
    losses.policy_loss /= static_cast<float>(games_n);
    losses.outcome_loss /= static_cast<float>(games_n);
  }

  void ResetTraining() {
    sgd_.reset(new nn::Adam(*primary_network_));
  }

 private:
  static internal::PlayoutConfig<GameSubclass> SelfPlayConfigToPlayoutConfig(
      const GameSubclass& game, const SelfPlayConfig& config) {
    return {
        .game = game,
        .n = config.playouts_n,
        .outcome_weight = config.outcome_weight,
        .policy_weight = config.policy_weight,
        .revisit_weight = config.revisit_weight,
        .policy_noise = config.policy_noise};
  }

  struct NetworkUpdateRow {
    // The inputs to the network.
    std::vector<nn::DynamicMatrix> state_input;

    // The outcome we should predict.
    nn::Matrix<GameSubclass::players_n(), 1> outcome_target;

    // The policy we should predict.
    nn::DynamicMatrix policy_target;

    // Which of the policy heads to update. If there is no policy, = -1.
    int policy_class_i;
  };

  TrainResult TrainIteration(
      internal::PlayoutConfig<GameSubclass>& playout_config, 
      float learning_rate) {
    std::vector<NetworkUpdateRow> updates;
    while (playout_config.game.State() == games::GameState::kOngoing) {
      internal::PlayoutResult<GameSubclass> result = playout_runner_->Playout(
          playout_config, *work_queue_);
      updates.emplace_back(
          std::move(playout_config.game.StateToMatrix()),
          std::move(result.outcome),
          std::move(result.policy),
          playout_config.game.PolicyClassI());
      playout_config.game.MakeMove(result.max_option_i);
    }

    auto losses = UpdatePrimaryModel(updates, learning_rate);
    std::vector<nn::DynamicMatrixRef> variables;
    primary_network_->GetVariables({}, variables);
    work_queue_->SetAllVariables(variables);
    playout_runner_->ClearModelOutputCache();

    return losses;
  }

  struct GradientResult {
    std::vector<nn::DynamicMatrix> var_grad;
    std::vector<uint32_t> var_grad_i;

    // This doesn't store loss averages, just the individual losses for a single
    // turn.
    TrainResult losses;
  };

  class GradientsWorkElement : 
      public internal::GameNetworkWorkItem<GameNetworkSubclass> {
   public:
    GradientsWorkElement(const GradientsWorkElement&) = delete;
    GradientsWorkElement& operator=(const GradientsWorkElement&) = delete;

    GradientsWorkElement(
        const NetworkUpdateRow& update, 
        GradientResult& grads) :
            update_(update), grads_(grads) {}

    void operator()(GameNetworkSubclass* local_network) const override {
      uint32_t policy_target_index = 
          local_network->policy_target_constant_indices()[
              update_.policy_class_i];
      uint32_t outcome_target_index = 
          local_network->outcome_target_constant_index();

      local_network->SetConstants(local_network->input_constant_indices(),
                                  update_.state_input);
      local_network->SetConstants({policy_target_index}, 
                                  {update_.policy_target});
      local_network->SetConstants({outcome_target_index},
                                  {update_.outcome_target});

      uint32_t policy_loss_index =
          local_network->policy_loss_target_indices()[
              update_.policy_class_i];
      uint32_t outcome_loss_index = local_network->outcome_loss_target_index();

      std::vector<float> losses;
      local_network->Gradients({policy_loss_index, outcome_loss_index}, 
                               grads_.var_grad_i, grads_.var_grad, losses);
      grads_.losses.policy_loss = losses[0];
      grads_.losses.outcome_loss = losses[1];
    }
   
   private:
    const NetworkUpdateRow& update_;
    GradientResult& grads_;
  };

  // Returns the loss averages across all simulated turns.
  TrainResult UpdatePrimaryModel(const std::vector<NetworkUpdateRow>& updates, 
                                 float lr) {
    if (updates.empty()) {
      LOG(FATAL) << "No model updates from iteration.";
    }

    std::vector<GradientResult> all_grads;
    all_grads.resize(updates.size());
    for (int i = 0; i < updates.size(); ++i) {
      work_queue_->AddWork(
          std::make_unique<GradientsWorkElement>(updates[i], all_grads[i]));
    }
    work_queue_->Drain();

    // Since not all variables are guaranteed to have gradients for each row in
    // updates, we have to do some extra bookkeeping to sum the terms of each
    // variable's average gradients.
    absl::flat_hash_map<uint32_t, int> var_index_to_vec_index;
    
    // Parallel arrays
    std::vector<nn::DynamicMatrix> acc_grads;
    std::vector<uint32_t> acc_index;
    std::vector<int> acc_grad_count;

    TrainResult losses{0.0f, 0.0f};

    for (auto& result : all_grads) {
      for (int i = 0; i < result.var_grad.size(); ++i) {
        uint32_t grad_index = result.var_grad_i[i];
        nn::DynamicMatrix& grad = result.var_grad[i];
        auto [iter, is_new] = var_index_to_vec_index.insert({grad_index,
                                                             acc_grads.size()});
        if (is_new) {
          acc_grads.push_back(std::move(grad));
          acc_index.push_back(grad_index);
          acc_grad_count.push_back(1);
        } else {
          acc_grads[iter->second] += grad;
          (acc_grad_count[iter->second])++;
        }
      }
      losses.policy_loss += result.losses.policy_loss;
      losses.outcome_loss += result.losses.outcome_loss;
    }

    for (int i = 0; i < acc_grads.size(); ++i) {
      acc_grads[i] /= static_cast<float>(acc_grad_count[i]);
    }
    losses.policy_loss /= static_cast<float>(all_grads.size());
    losses.outcome_loss /= static_cast<float>(all_grads.size());

    sgd_->Update(lr, acc_index, acc_grads, *primary_network_);

    return losses;
  }

  std::unique_ptr<GameNetworkSubclass> primary_network_;
  std::unique_ptr<nn::Adam> sgd_;

  std::unique_ptr<DispatchQueue> work_queue_;
  std::unique_ptr<PlayoutRunner> playout_runner_;
};

}  // namespace mcts
}  // namespace azah

#endif  // AZAH_MCTS_SELF_PLAYER_H_
 