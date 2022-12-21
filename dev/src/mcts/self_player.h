#ifndef AZAH_MCTS_SELF_PLAYER_H_
#define AZAH_MCTS_SELF_PLAYER_H_

#include <array>
#include <memory>
#include <vector>

#include "../mcts/network_dispatch_queue.h"
#include "../mcts/playout_runner.h"
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
      work_queue_(new DispatchQueue(threads, DispatchQueueLength)), 
      playout_runner_(new PlayoutRunner()) {}

  SelfPlayer(
      int threads, 
      const GameSubclass& init_game, 
      const GameNetworkSubclass& init_network) :
          game_(init_game),
          work_queue_(new DispatchQueue(threads, DispatchQueueLength)),
          playout_runner_(new PlayoutRunner()) {
    set_network_variables(init_network);
  }

  void set_game(const GameSubclass& game) {
    game_ = game;
  }

  void set_network_variables(const GameNetworkSubclass& network) {
    std::vector<nn::ConstDynamicMatrixRef> variables;
    network.GetVariables({}, variables);
    primary_network_.SetVariables({}, variables);
  }

  const GameNetworkSubclass& get_network() const {
    return primary_network_;
  }

  EvaluateResult<GameSubclass> EvaluatePosition(const SelfPlayConfig& config) {
    //
    //
    //
  }

  void Train(int games_n, const SelfPlayConfig& config) {
    //
    //
    //
  }

 private:
  GameSubclass game_;
  GameNetworkSubclass primary_network_;

  std::unique_ptr<DispatchQueue> work_queue_;
  std::unique_ptr<PlayoutRunner> playout_runner_;
};

}  // namespace mcts
}  // namespace azah

#endif  // AZAH_MCTS_SELF_PLAYER_H_
