#ifndef AZAH_GAMES_IGNOBLE_IGNOBLE4_NETWORK_H_
#define AZAH_GAMES_IGNOBLE_IGNOBLE4_NETWORK_H_

#include "../../nn/constant.h"
#include "../../nn/op/broadcast_add.h"
#include "../../nn/op/broadcast_matmul.h"
#include "../../nn/op/concat_cols.h"
#include "../../nn/op/matmul.h"
#include "../../nn/op/mixer.h"
#include "../../nn/op/row_mean.h"
#include "../../nn/op/softmax.h"
#include "../../nn/op/softmax_cross_ent.h"
#include "../../nn/variable.h"
#include "../game_network.h"

namespace azah {
namespace games {
namespace ignoble {

class Ignoble4Network : public GameNetwork {
 public:
  Ignoble4Network(const Ignoble4Network&) = delete;
  Ignoble4Network& operator=(const Ignoble4Network&) = delete;

  Ignoble4Network();

 private:
  static constexpr int kFeatureDepth = 64;
 
  // Four inputs for each board, with the first position being the current
  // player's turn and the other's rotated accordingly.
  nn::Constant<129, 1> input_pos_1_;
  nn::Constant<129, 1> input_pos_2_;
  nn::Constant<129, 1> input_pos_3_;
  nn::Constant<129, 1> input_pos_4_;
  nn::Constant<60, 1> input_global_;

  // We expand each input (non-global) into 4 kFeatureDepth vectors.
  nn::Variable<256, 129> input_embedding_k_;
  nn::op::BroadcastMatmul<256, 129, 4> input_embedding_;

  // Three concats to get a 64x16 column vector from 4 64x4s. 
  nn::op::ConcatCols<64, 4, 4> concat_1_;
  nn::op::ConcatCols<64, 4, 4> concat_2_;
  nn::op::ConcatCols<64, 8, 8> concat_3_;

  // We expand the global state into a 64-D vector added to all of the columns.
  nn::Variable<64, 60> input_global_embedding_k_;
  nn::op::Matmul<64, 60, 60, 1> input_global_embedding_;

  // Add the global embedding to the input embeddings.
  nn::op::BroadcastAdd<64, 16> global_to_features_;

  // At this, we have a 64x16 state matrix that we can operate on normally.

  static constexpr int kMixerTokenDepth = 64;
  static constexpr int kMixerFeatureDepth = 256;

  // 3x MLP-Mixer: https://arxiv.org/pdf/2105.01601.pdf
  nn::op::Mixer<kFeatureDepth, 16, kMixerTokenDepth, kMixerFeatureDepth> mix_1_;
  nn::op::Mixer<kFeatureDepth, 16, kMixerTokenDepth, kMixerFeatureDepth> mix_2_;
  nn::op::Mixer<kFeatureDepth, 16, kMixerTokenDepth, kMixerFeatureDepth> mix_3_;

  // One final norm and average pool.
  nn::op::LayerNorm<kFeatureDepth, 16> final_norm_;
  nn::op::RowMean<kFeatureDepth, 16> pool_;

  // Fork final layer to outputs

  // This little optimization only works because we'll always backprop two
  // losses from the model at a time: the outcome and policy. So we can have as
  // many heads as we want, as long as this forks twice.
  nn::op::Fork<kFeatureDepth, 1> pool_fork_;

  // Policy heads

  //
  // Implement 'em all!
  //

  nn::Variable<6, kFeatureDepth> policy_linear_k_;
  nn::op::Matmul<6, kFeatureDepth, kFeatureDepth, 1> policy_linear_;
  nn::op::Softmax<6, 1> policy_;

  nn::Constant<6, 1> policy_target_;
  nn::op::SoftmaxCrossEnt<6, 1> policy_loss_;

  // Outcome head

  nn::Variable<4, kFeatureDepth> outcome_linear_k_;
  nn::op::Matmul<4, kFeatureDepth, kFeatureDepth, 1> outcome_linear_;
  nn::op::Softmax<4, 1> outcome_;

  nn::Constant<4, 1> outcome_target_;
  nn::op::SoftmaxCrossEnt<4, 1> outcome_loss_;
};

}  // namespace ignoble
}  // namespace games
}  // namespace azah

#endif  // AZAH_GAMES_IGNOBLE_IGNOBLE4_NETWORK_H_
