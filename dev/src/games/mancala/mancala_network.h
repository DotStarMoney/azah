#ifndef AZAH_GAMES_MANCALA_MANCALA_NETWORK_H_
#define AZAH_GAMES_MANCALA_MANCALA_NETWORK_H_

#include "../../nn/constant.h"
#include "../../nn/init.h"
#include "../../nn/network.h"
#include "../../nn/op/fork.h"
#include "../../nn/op/layer_norm.h"
#include "../../nn/op/matmul.h"
#include "../../nn/op/mixer.h"
#include "../../nn/op/row_mean.h"
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
  
  static constexpr int kMixerTokenDepth = 64;
  static constexpr int kMixerFeatureDepth = 256;

  // Board must be rotated such that the current player's pockets are along the
  // first 7 cols.
  nn::Constant<48, 14> input_;

  // Start by applying the same linear transformation to each 48-D one-hot
  // input.
  nn::Variable<kFeatureDepth, 48> input_embedding_k_;
  nn::op::Matmul<kFeatureDepth, 48, 48, 14> input_embedding_;

  // 3x MLP-Mixer: https://arxiv.org/pdf/2105.01601.pdf
  nn::op::Mixer<kFeatureDepth, 14, kMixerTokenDepth, kMixerFeatureDepth> mix_1_;
  nn::op::Mixer<kFeatureDepth, 14, kMixerTokenDepth, kMixerFeatureDepth> mix_2_;
  nn::op::Mixer<kFeatureDepth, 14, kMixerTokenDepth, kMixerFeatureDepth> mix_3_;

  // One final norm and average pool.
  nn::Variable<kFeatureDepth, 1> final_norm_beta_;
  nn::Variable<kFeatureDepth, 1> final_norm_gamma_;
  nn::op::LayerNorm<kFeatureDepth, 14> final_norm_;
  nn::op::RowMean<kFeatureDepth, 14> pool_;

  // Fork final layer to outputs

  // This little optimization only works because we'll always backprop two
  // losses from the model at a time: the outcome and policy.
  nn::op::Fork<kFeatureDepth, 1> pool_fork_;

  // Policy head

  nn::Variable<6, kFeatureDepth> policy_linear_k_;
  nn::op::Matmul<6, kFeatureDepth, kFeatureDepth, 1> policy_linear_;
  nn::op::Softmax<6, 1> policy_;

  nn::Constant<6, 1> policy_target_;
  nn::op::SoftmaxCrossEnt<6, 1> policy_loss_;

  // Outcome head

  nn::Variable<2, kFeatureDepth> outcome_linear_k_;
  nn::op::Matmul<2, kFeatureDepth, kFeatureDepth, 1> outcome_linear_;
  nn::op::Softmax<2, 1> outcome_;

  nn::Constant<2, 1> outcome_target_;
  nn::op::SoftmaxCrossEnt<2, 1> outcome_loss_;
};

}  // namespace mancala
}  // namespace games
}  // namespace azah

#endif  // AZAH_GAMES_MANCALA_MANCALA_NETWORK_H_
