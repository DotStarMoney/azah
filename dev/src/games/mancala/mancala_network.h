#ifndef AZAH_GAMES_MANCALA_MANCALA_NETWORK_H_
#define AZAH_GAMES_MANCALA_MANCALA_NETWORK_H_

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
namespace mancala {

class MancalaNetwork : public GameNetwork {
 public:
  MancalaNetwork(const MancalaNetwork&) = delete;
  MancalaNetwork& operator=(const MancalaNetwork&) = delete;

  MancalaNetwork();

 private:
  static constexpr int kFeatureDepth = 64;

  nn::Constant<48, 14> input_;

  nn::Variable<kFeatureDepth, 48> input_embedding_k_;
  nn::op::Matmul<kFeatureDepth, 48, 48, 14> input_embedding_;

  /*


  // Fork final layer to outputs

  // This little optimization only works because we'll always backprop two
  // losses from the model at a time: the outcome and policy.
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
  */
};

}  // namespace mancala
}  // namespace games
}  // namespace azah

#endif  // AZAH_GAMES_MANCALA_MANCALA_NETWORK_H_
