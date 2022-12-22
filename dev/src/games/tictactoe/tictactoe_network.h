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

  azah::nn::Variable<kLayer1Depth, 9> dense1_k_;
  azah::nn::op::Matmul<kLayer1Depth, 9, 9, 1> dense1_;

  azah::nn::Variable<kLayer1Depth, 1> norm1_b_;
  azah::nn::Variable<kLayer1Depth, 1> norm1_g_;
  azah::nn::op::LayerNorm<kLayer1Depth, 1> norm1_;

  azah::nn::op::Swish<kLayer1Depth, 1> swish1_;

  // Layer 2

  azah::nn::Variable<kLayer2Depth, kLayer1Depth> dense2_k_;
  azah::nn::op::Matmul<kLayer2Depth, kLayer1Depth, kLayer1Depth, 1> dense2_;

  azah::nn::Variable<kLayer2Depth, 1> norm2_b_;
  azah::nn::Variable<kLayer2Depth, 1> norm2_g_;
  azah::nn::op::LayerNorm<kLayer2Depth, 1> norm2_;

  azah::nn::op::Swish<kLayer2Depth, 1> swish2_;

  // Fork final layer to outputs

  azah::nn::op::Fork<kLayer2Depth, 1> swish2_fork_;

  // Policy head

  azah::nn::Variable<9, kLayer2Depth> policy_linear_k_;
  azah::nn::op::Matmul<9, kLayer2Depth, kLayer2Depth, 1> policy_linear_;
  azah::nn::op::Softmax<9, 1> policy_;
  
  nn::Constant<9, 1> policy_target_;
  azah::nn::op::SoftmaxCrossEnt<9, 1> policy_loss_;

  // Outcome head

  azah::nn::Variable<2, kLayer2Depth> outcome_linear_k_;
  azah::nn::op::Matmul<2, kLayer2Depth, kLayer2Depth, 1> outcome_linear_;
  azah::nn::op::Softmax<2, 1> outcome_;

  nn::Constant<2, 1> outcome_target_;
  azah::nn::op::SoftmaxCrossEnt<2, 1> outcome_loss_;
};

}  // namespace tictactoe
}  // namespace games
}  // namespace azah

#endif  // AZAH_GAMES_TICTACTOE_TICTACTOE_NETWORK_H_
