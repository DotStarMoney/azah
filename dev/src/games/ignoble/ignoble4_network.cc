#include "ignoble4_network.h"

#include "../../nn/init.h"
#include "../game_network.h"

namespace azah {
namespace games {
namespace ignoble {
  
Ignoble4Network::Ignoble4Network() :
    GameNetwork(
        {0, 1, 2, 3, 4}, 
        {5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, 
        15, 
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 
        10, 
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 
        10, 
        {4, 16, 4, 3, 4, 2, 2, 4, 8, 4}),
    input_pos_1_(nn::init::Zeros<129, 1>()),
    input_pos_2_(nn::init::Zeros<129, 1>()),
    input_pos_3_(nn::init::Zeros<129, 1>()),
    input_pos_4_(nn::init::Zeros<129, 1>()),
    input_global_(nn::init::Zeros<60, 1>()),
    input_embedding_k_(nn::init::GlorotUniform<256, 129, 64, 129>()),
    input_embedding_pos_1_(input_embedding_k_, input_pos_1_),
    input_embedding_pos_2_(input_embedding_k_, input_pos_2_),
    input_embedding_pos_3_(input_embedding_k_, input_pos_3_),
    input_embedding_pos_4_(input_embedding_k_, input_pos_4_),
    concat_1_(input_embedding_pos_1_, input_embedding_pos_2_),
    concat_2_(input_embedding_pos_3_, input_embedding_pos_4_),
    concat_3_(concat_1_, concat_2_),
    input_global_embedding_k_(nn::init::GlorotUniform<64, 60>()),
    input_global_embedding_(input_global_embedding_k_, input_global_),
    global_to_features_(input_global_embedding_, concat_3_),
    mix_1_(global_to_features_),
    // This cast is to prevent this being interpreted as a copy.
    mix_2_(static_cast<nn::Node<64, 16>&>(mix_1_)),
    final_norm_(mix_2_),
    pool_(final_norm_),
    pool_fork_(pool_, 2),
    p_team_select_linear_k_(nn::init::GlorotUniform<4, kFeatureDepth>()),
    p_team_select_linear_(p_team_select_linear_k_, pool_fork_),
    p_team_select_(p_team_select_linear_),
    p_team_select_target_(nn::init::Zeros<4, 1>()),
    p_team_select_loss_(p_team_select_linear_, p_team_select_target_),
    p_character_select_linear_k_(nn::init::GlorotUniform<16, kFeatureDepth>()),
    p_character_select_linear_(p_character_select_linear_k_, pool_fork_),
    p_character_select_(p_character_select_linear_),
    p_character_select_target_(nn::init::Zeros<16, 1>()),
    p_character_select_loss_(p_character_select_linear_,
                             p_character_select_target_),
    p_princess_stock_linear_k_(nn::init::GlorotUniform<4, kFeatureDepth>()),
    p_princess_stock_linear_(p_princess_stock_linear_k_, pool_fork_),
    p_princess_stock_(p_princess_stock_linear_),
    p_princess_stock_target_(nn::init::Zeros<4, 1>()),
    p_princess_stock_loss_(p_princess_stock_linear_, p_princess_stock_target_),
    p_meat_bungler_bounty_linear_k_(
        nn::init::GlorotUniform<3, kFeatureDepth>()),
    p_meat_bungler_bounty_linear_(p_meat_bungler_bounty_linear_k_, pool_fork_),
    p_meat_bungler_bounty_(p_meat_bungler_bounty_linear_),
    p_meat_bungler_bounty_target_(nn::init::Zeros<3, 1>()),
    p_meat_bungler_bounty_loss_(p_meat_bungler_bounty_linear_, 
                                p_meat_bungler_bounty_target_),
    p_merry_pieman_stock_linear_k_(nn::init::GlorotUniform<4, kFeatureDepth>()),
    p_merry_pieman_stock_linear_(p_merry_pieman_stock_linear_k_, pool_fork_),
    p_merry_pieman_stock_(p_merry_pieman_stock_linear_),
    p_merry_pieman_stock_target_(nn::init::Zeros<4, 1>()),
    p_merry_pieman_stock_loss_(p_merry_pieman_stock_linear_, 
                               p_merry_pieman_stock_target_),
    p_benedict_increase_linear_k_(nn::init::GlorotUniform<2, kFeatureDepth>()),
    p_benedict_increase_linear_(p_benedict_increase_linear_k_, pool_fork_),
    p_benedict_increase_(p_benedict_increase_linear_),
    p_benedict_increase_target_(nn::init::Zeros<2, 1>()),
    p_benedict_increase_loss_(p_benedict_increase_linear_, 
                              p_benedict_increase_target_),
    p_bethesda_swap_linear_k_(nn::init::GlorotUniform<2, kFeatureDepth>()),
    p_bethesda_swap_linear_(p_bethesda_swap_linear_k_, pool_fork_),
    p_bethesda_swap_(p_bethesda_swap_linear_),
    p_bethesda_swap_target_(nn::init::Zeros<2, 1>()),
    p_bethesda_swap_loss_(p_bethesda_swap_linear_, p_bethesda_swap_target_),
    p_ounce_steal_stock_linear_k_(nn::init::GlorotUniform<4, kFeatureDepth>()),
    p_ounce_steal_stock_linear_(p_ounce_steal_stock_linear_k_, pool_fork_),
    p_ounce_steal_stock_(p_ounce_steal_stock_linear_),
    p_ounce_steal_stock_target_(nn::init::Zeros<4, 1>()),
    p_ounce_steal_stock_loss_(p_ounce_steal_stock_linear_, 
                              p_ounce_steal_stock_target_),
    p_magician_stock_take_toss_linear_k_(
        nn::init::GlorotUniform<8, kFeatureDepth>()),
    p_magician_stock_take_toss_linear_(p_magician_stock_take_toss_linear_k_, 
                                       pool_fork_),
    p_magician_stock_take_toss_(p_magician_stock_take_toss_linear_),
    p_magician_stock_take_toss_target_(nn::init::Zeros<8, 1>()),
    p_magician_stock_take_toss_loss_(p_magician_stock_take_toss_linear_,
                                     p_magician_stock_take_toss_target_),
    p_repent_stock_linear_k_(nn::init::GlorotUniform<4, kFeatureDepth>()),
    p_repent_stock_linear_(p_repent_stock_linear_k_, pool_fork_),
    p_repent_stock_(p_repent_stock_linear_),
    p_repent_stock_target_(nn::init::Zeros<4, 1>()),
    p_repent_stock_loss_(p_repent_stock_linear_, p_repent_stock_target_),
    outcome_linear_k_(nn::init::GlorotUniform<4, kFeatureDepth>()),
    outcome_linear_(outcome_linear_k_, pool_fork_),
    outcome_(outcome_linear_),
    outcome_target_(nn::init::Zeros<4, 1>()),
    outcome_loss_(outcome_linear_, outcome_target_) {
  AddOutput(&p_team_select_);
  AddOutput(&p_character_select_);
  AddOutput(&p_princess_stock_);
  AddOutput(&p_meat_bungler_bounty_);
  AddOutput(&p_merry_pieman_stock_);
  AddOutput(&p_benedict_increase_);
  AddOutput(&p_bethesda_swap_);
  AddOutput(&p_ounce_steal_stock_);
  AddOutput(&p_magician_stock_take_toss_);
  AddOutput(&p_repent_stock_);
  AddOutput(&outcome_);

  AddTarget(&p_team_select_loss_);
  AddTarget(&p_character_select_loss_);
  AddTarget(&p_princess_stock_loss_);
  AddTarget(&p_meat_bungler_bounty_loss_);
  AddTarget(&p_merry_pieman_stock_loss_);
  AddTarget(&p_benedict_increase_loss_);
  AddTarget(&p_bethesda_swap_loss_);
  AddTarget(&p_ounce_steal_stock_loss_);
  AddTarget(&p_magician_stock_take_toss_loss_);
  AddTarget(&p_repent_stock_loss_);
  AddTarget(&outcome_loss_);

  AddVariable(&input_embedding_k_);
  AddVariable(&input_global_embedding_k_);
  AddVariables(mix_1_);
  AddVariables(mix_2_);
  AddVariables(final_norm_);
  AddVariable(&p_team_select_linear_k_);
  AddVariable(&p_character_select_linear_k_);
  AddVariable(&p_princess_stock_linear_k_);
  AddVariable(&p_meat_bungler_bounty_linear_k_);
  AddVariable(&p_merry_pieman_stock_linear_k_);
  AddVariable(&p_benedict_increase_linear_k_);
  AddVariable(&p_bethesda_swap_linear_k_);
  AddVariable(&p_ounce_steal_stock_linear_k_);
  AddVariable(&p_magician_stock_take_toss_linear_k_);
  AddVariable(&p_repent_stock_linear_k_);
  AddVariable(&outcome_linear_k_);

  AddConstant(&input_pos_1_);
  AddConstant(&input_pos_2_);
  AddConstant(&input_pos_3_);
  AddConstant(&input_pos_4_);
  AddConstant(&input_global_);
  AddConstant(&p_team_select_target_);
  AddConstant(&p_character_select_target_);
  AddConstant(&p_princess_stock_target_);
  AddConstant(&p_meat_bungler_bounty_target_);
  AddConstant(&p_merry_pieman_stock_target_);
  AddConstant(&p_benedict_increase_target_);
  AddConstant(&p_bethesda_swap_target_);
  AddConstant(&p_ounce_steal_stock_target_);
  AddConstant(&p_magician_stock_take_toss_target_);
  AddConstant(&p_repent_stock_target_);
  AddConstant(&outcome_target_);
}

}  // namespace ignoble
}  // namespace games
}  // namespace azah
