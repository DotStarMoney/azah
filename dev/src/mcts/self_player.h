#ifndef AZAH_MCTS_SELF_PLAYER_H_
#define AZAH_MCTS_SELF_PLAYER_H_

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

#include "network_dispatch_queue.h"
#include "game_network_work_item.h"
#include "glog/logging.h"
#include "playout_runner.h"
#include "../nn/adam.h"
#include "../nn/data_types.h"

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
      LOG(FATAL) << "Game is over."
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

  void Train(int games_n, const SelfPlayConfig& config) {
    auto playout_config = SelfPlayConfigToPlayoutConfig(GameSubclass(), config);
    for (int i = 0; i < games_n; ++i) {
      TrainIteration(playout_config, config.learning_rate);
    }
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

  void TrainIteration(
      const internal::PlayoutConfig<GameSubclass>& playout_config, 
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

    UpdateModel(updates, learning_rate);
    std::vector<nn::DynamicMatrixRef> variables;
    primary_network_->GetVariables({}, variables);
    work_queue_->SetAllVariables(variables);
    playout_runner_->ClearModelOutputCache();
  }

  class GradientsWorkElement : 
      public internal::GameNetworkWorkItem<GameNetworkSubclass> {
   public:
    GradientsWorkElement(const GradientsWorkElement&) = delete;
    GradientsWorkElement& operator=(const GradientsWorkElement&) = delete;

    GradientsWorkElement(
        const NetworkUpdateRow& update, 
        std::vector<nn::DynamicMatrix>& grads) : 
            update_(update), grads_(grads) {}

    void operator()(GameNetworkSubclass* local_network) {
      //
      // Get some gradients I guess, put 'em in grads_
      //
    }
   
   private:
    const NetworkUpdateRow& update_;
    std::vector<nn::DynamicMatrix>& grads_;
  };

  void UpdateModel(const std::vector<NetworkUpdateRow>& updates, float lr) {
    //
    // Needs to account for the fact that some updates will yield incomplete
    // gradients.
    //

    /*
    if (updates.empty()) {
      LOG(FATAL) << "No model updates from iteration.";
    }

    std::vector<std::vector<nn::DynamicMatrix>> all_grads;
    all_grads.resize(updates.size());
    for (int i = 0; i < updates.size(); ++i) {
      work_queue_->AddWork(
          std::make_unique<GradientsWorkElement>(updates[i], all_grads[i]));
    }
    work_queue_->Drain();

    for (int var_i = 0; var_i < all_grads[0].size(); ++var_i) {
      for (int replica_i = 0; replica_i < all_grads.size(); ++replica_i) {
        all_grads[0][var_i] += all_grads[replica_i][var_i];
      }
      all_grads[0][var_i] /= static_cast<float>(all_grads.size());
    }

    sgd_->Update(lr, {}, all_grads[0], *primary_network_);
    */
  }

  std::unique_ptr<GameNetworkSubclass> primary_network_;
  std::unique_ptr<nn::Adam> sgd_;

  std::unique_ptr<DispatchQueue> work_queue_;
  std::unique_ptr<PlayoutRunner> playout_runner_;
};

}  // namespace mcts
}  // namespace azah

#endif  // AZAH_MCTS_SELF_PLAYER_H_
 