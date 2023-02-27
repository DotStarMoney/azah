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
    //
    // Wire it uhp.
    //
    input_(nn::init::Zeros<48, 14>()),
    input_embedding_k_(nn::init::GlorotUniform<kFeatureDepth, 48>()),
    input_embedding_(input_embedding_k_, input_),
    mix_1_(input_embedding_),
    mix_2_(input_embedding_),
    final_norm_(mix_2_),
    pool_(final_norm_),
    pool_fork_(pool_, 2),
    policy_linear_k_(nn::init::GlorotUniform<6, kFeatureDepth>()),
    policy_linear_(policy_linear_k_, pool_fork_),
    policy_(policy_linear_),
    policy_target_(nn::init::Zeros<6, 1>()),
    policy_loss_(policy_linear_, policy_target_),
    outcome_linear_k_(nn::init::GlorotUniform<2, kFeatureDepth>()),
    outcome_linear_(outcome_linear_k_, pool_fork_),
    outcome_(outcome_linear_),
    outcome_target_(nn::init::Zeros<2, 1>()),
    outcome_loss_(outcome_linear_, outcome_target_) {
  AddOutput(&policy_);
  AddOutput(&outcome_);

  AddTarget(&policy_loss_);
  AddTarget(&outcome_loss_);

  AddVariable(&input_embedding_k_);
  AddVariables(mix_1_);
  AddVariables(mix_2_);
  AddVariables(final_norm_);
  AddVariable(&policy_linear_k_);
  AddVariable(&outcome_linear_k_);

  AddConstant(&input_);
  AddConstant(&policy_target_);
  AddConstant(&outcome_target_);
}

}  // namespace ignoble
}  // namespace games
}  // namespace azah
