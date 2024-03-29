#ifndef AZAH_GAMES_TICTACTOE_TICTACTOE_NETWORK_H_
#define AZAH_GAMES_TICTACTOE_TICTACTOE_NETWORK_H_

#include "../../nn/constant.h"
#include "../../nn/init.h"
#include "../../nn/network.h"
#include "../../nn/op/fork.h"
#include "../../nn/op/layer_norm.h"
#include "../../nn/op/matmul.h"
#include "../../nn/op/softmax.h"
#include "../../nn/op/softmax_cross_ent.h"
#include "../../nn/op/swish.h"
#include "../../nn/variable.h"
#include "../game_network.h"

namespace azah {
namespace games {
namespace tictactoe {

class TictactoeNetwork : public GameNetwork {
 public:
  TictactoeNetwork(const TictactoeNetwork&) = delete;
  TictactoeNetwork& operator=(const TictactoeNetwork&) = delete;

  TictactoeNetwork();

 private:
  static constexpr int kLayer1Depth = 32;
  static constexpr int kLayer2Depth = 32;

  // HW flattened board. 1 = P1, 0 = N/A, -1 = P2.
  nn::Constant<9, 1> input_;

  // Layer 1

  nn::Variable<kLayer1Depth, 9> dense1_k_;
  nn::op::Matmul<kLayer1Depth, 9, 9, 1> dense1_;

  nn::op::LayerNorm<kLayer1Depth, 1> norm1_;
  nn::op::Swish<kLayer1Depth, 1> swish1_;

  // Layer 2

  nn::Variable<kLayer2Depth, kLayer1Depth> dense2_k_;
  nn::op::Matmul<kLayer2Depth, kLayer1Depth, kLayer1Depth, 1> dense2_;

  nn::op::LayerNorm<kLayer2Depth, 1> norm2_;
  nn::op::Swish<kLayer2Depth, 1> swish2_;

  // Fork final layer to outputs

  // This little optimization only works because we'll always backprop two
  // losses from the model at a time: the outcome and policy.
  nn::op::Fork<kLayer2Depth, 1> swish2_fork_;

  // Policy head

  nn::Variable<9, kLayer2Depth> policy_linear_k_;
  nn::op::Matmul<9, kLayer2Depth, kLayer2Depth, 1> policy_linear_;
  nn::op::Softmax<9, 1> policy_;
  
  nn::Constant<9, 1> policy_target_;
  nn::op::SoftmaxCrossEnt<9, 1> policy_loss_;

  // Outcome head

  nn::Variable<2, kLayer2Depth> outcome_linear_k_;
  nn::op::Matmul<2, kLayer2Depth, kLayer2Depth, 1> outcome_linear_;
  nn::op::Softmax<2, 1> outcome_;

  nn::Constant<2, 1> outcome_target_;
  nn::op::SoftmaxCrossEnt<2, 1> outcome_loss_;
};

}  // namespace tictactoe
}  // namespace games
}  // namespace azah

#endif  // AZAH_GAMES_TICTACTOE_TICTACTOE_NETWORK_H_
