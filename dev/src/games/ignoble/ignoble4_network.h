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
  static constexpr int kFeatureDepth = 80;
  static constexpr int kExpandFeatureDepth = kFeatureDepth * 4;
 
  // Four inputs for each board, with the first position being the current
  // player's turn and the other's rotated accordingly.
  nn::Constant<129, 1> input_pos_1_;
  nn::Constant<129, 1> input_pos_2_;
  nn::Constant<129, 1> input_pos_3_;
  nn::Constant<129, 1> input_pos_4_;
  nn::Constant<60, 1> input_global_;

  // We expand each input (non-global) into 4 kFeatureDepth vectors.
  nn::Variable<kExpandFeatureDepth, 129> input_embedding_k_;

  nn::op::BroadcastMatmul<kExpandFeatureDepth, 129, 4> input_embedding_pos_1_;
  nn::op::BroadcastMatmul<kExpandFeatureDepth, 129, 4> input_embedding_pos_2_;
  nn::op::BroadcastMatmul<kExpandFeatureDepth, 129, 4> input_embedding_pos_3_;
  nn::op::BroadcastMatmul<kExpandFeatureDepth, 129, 4> input_embedding_pos_4_;

  // Three concats to get a 64x16 column vector from 4 64x4s. 
  nn::op::ConcatCols<kFeatureDepth, 4, 4> concat_1_;
  nn::op::ConcatCols<kFeatureDepth, 4, 4> concat_2_;
  nn::op::ConcatCols<kFeatureDepth, 8, 8> concat_3_;

  // We expand the global state into a 64-D vector added to all of the columns.
  nn::Variable<kFeatureDepth, 60> input_global_embedding_k_;
  nn::op::Matmul<kFeatureDepth, 60, 60, 1> input_global_embedding_;

  // Add the global embedding to the input embeddings.
  nn::op::BroadcastAdd<kFeatureDepth, 16> global_to_features_;

  // At this, we have a 64x16 state matrix that we can operate on normally.

  static constexpr int kMixerTokenDepth = 64;
  static constexpr int kMixerFeatureDepth = 256;

  // 2x MLP-Mixer: https://arxiv.org/pdf/2105.01601.pdf
  nn::op::Mixer<kFeatureDepth, 16, kMixerTokenDepth, kMixerFeatureDepth> mix_1_;
  nn::op::Mixer<kFeatureDepth, 16, kMixerTokenDepth, kMixerFeatureDepth> mix_2_;

  // One final norm and average pool.
  nn::op::LayerNorm<kFeatureDepth, 16> final_norm_;
  nn::op::RowMean<kFeatureDepth, 16> pool_;

  // Fork final layer to outputs

  // This little optimization only works because we'll always backprop two
  // losses from the model at a time: the outcome and policy. So we can have as
  // many heads as we want, as long as this forks twice.
  nn::op::Fork<kFeatureDepth, 1> pool_fork_;

  // Policy heads

  // 1) TeamSelect

  nn::Variable<4, kFeatureDepth> p_team_select_linear_k_;
  nn::op::Matmul<4, kFeatureDepth, kFeatureDepth, 1> p_team_select_linear_;
  nn::op::Softmax<4, 1> p_team_select_;

  nn::Constant<4, 1> p_team_select_target_;
  nn::op::SoftmaxCrossEnt<4, 1> p_team_select_loss_;

  // 2) CharacterSelect

  nn::Variable<16, kFeatureDepth> p_character_select_linear_k_;
  nn::op::Matmul<16, kFeatureDepth, kFeatureDepth, 1> 
      p_character_select_linear_;
  nn::op::Softmax<16, 1> p_character_select_;

  nn::Constant<16, 1> p_character_select_target_;
  nn::op::SoftmaxCrossEnt<16, 1> p_character_select_loss_;

  // 3) PrincessStock

  nn::Variable<4, kFeatureDepth> p_princess_stock_linear_k_;
  nn::op::Matmul<4, kFeatureDepth, kFeatureDepth, 1> p_princess_stock_linear_;
  nn::op::Softmax<4, 1> p_princess_stock_;

  nn::Constant<4, 1> p_princess_stock_target_;
  nn::op::SoftmaxCrossEnt<4, 1> p_princess_stock_loss_;

  // 4) MeatBunglerToss

  nn::Variable<2, kFeatureDepth> p_meat_bungler_toss_linear_k_;
  nn::op::Matmul<2, kFeatureDepth, kFeatureDepth, 1> 
      p_meat_bungler_toss_linear_;
  nn::op::Softmax<2, 1> p_meat_bungler_toss_;

  nn::Constant<2, 1> p_meat_bungler_toss_target_;
  nn::op::SoftmaxCrossEnt<2, 1> p_meat_bungler_toss_loss_;

  // 5) MeatBunglerStock

  nn::Variable<2, kFeatureDepth> p_meat_bungler_stock_linear_k_;
  nn::op::Matmul<2, kFeatureDepth, kFeatureDepth, 1>
      p_meat_bungler_stock_linear_;
  nn::op::Softmax<2, 1> p_meat_bungler_stock_;

  nn::Constant<2, 1> p_meat_bungler_stock_target_;
  nn::op::SoftmaxCrossEnt<2, 1> p_meat_bungler_stock_loss_;

  // 6) MerryPiemanStock

  nn::Variable<4, kFeatureDepth> p_merry_pieman_stock_linear_k_;
  nn::op::Matmul<4, kFeatureDepth, kFeatureDepth, 1>
      p_merry_pieman_stock_linear_;
  nn::op::Softmax<4, 1> p_merry_pieman_stock_;

  nn::Constant<4, 1> p_merry_pieman_stock_target_;
  nn::op::SoftmaxCrossEnt<4, 1> p_merry_pieman_stock_loss_;

  // 7) BenedictIncrease

  nn::Variable<2, kFeatureDepth> p_benedict_increase_linear_k_;
  nn::op::Matmul<2, kFeatureDepth, kFeatureDepth, 1>
      p_benedict_increase_linear_;
  nn::op::Softmax<2, 1> p_benedict_increase_;

  nn::Constant<2, 1> p_benedict_increase_target_;
  nn::op::SoftmaxCrossEnt<2, 1> p_benedict_increase_loss_;

  // 8) BethesdaSwap

  nn::Variable<2, kFeatureDepth> p_bethesda_swap_linear_k_;
  nn::op::Matmul<2, kFeatureDepth, kFeatureDepth, 1> p_bethesda_swap_linear_;
  nn::op::Softmax<2, 1> p_bethesda_swap_;

  nn::Constant<2, 1> p_bethesda_swap_target_;
  nn::op::SoftmaxCrossEnt<2, 1> p_bethesda_swap_loss_;

  // 9) OunceStealStock

  nn::Variable<4, kFeatureDepth> p_ounce_steal_stock_linear_k_;
  nn::op::Matmul<4, kFeatureDepth, kFeatureDepth, 1> 
      p_ounce_steal_stock_linear_;
  nn::op::Softmax<4, 1> p_ounce_steal_stock_;

  nn::Constant<4, 1> p_ounce_steal_stock_target_;
  nn::op::SoftmaxCrossEnt<4, 1> p_ounce_steal_stock_loss_;

  // 10) MagicianStockTakeToss

  nn::Variable<8, kFeatureDepth> p_magician_stock_take_toss_linear_k_;
  nn::op::Matmul<8, kFeatureDepth, kFeatureDepth, 1>
      p_magician_stock_take_toss_linear_;
  nn::op::Softmax<8, 1> p_magician_stock_take_toss_;

  nn::Constant<8, 1> p_magician_stock_take_toss_target_;
  nn::op::SoftmaxCrossEnt<8, 1> p_magician_stock_take_toss_loss_;

  // 11) RepentStock

  nn::Variable<5, kFeatureDepth> p_repent_stock_linear_k_;
  nn::op::Matmul<5, kFeatureDepth, kFeatureDepth, 1> p_repent_stock_linear_;
  nn::op::Softmax<5, 1> p_repent_stock_;

  nn::Constant<5, 1> p_repent_stock_target_;
  nn::op::SoftmaxCrossEnt<5, 1> p_repent_stock_loss_;

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
