#include "mancala_network.h"

#include "../../nn/init.h"
#include "../game_network.h"

namespace azah {
namespace games {
namespace mancala {

MancalaNetwork::MancalaNetwork() :
    GameNetwork({0}, {1}, 2, {0}, 1, {0}, 1),
    input_(nn::init::Zeros<48, 14>()),
    input_embedding_k_(nn::init::GlorotUniform<kFeatureDepth, 48>()),
    input_embedding_(input_embedding_k_, input_),
    mix_(input_embedding_),
    final_norm_(mix_),
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
  AddVariables(mix_);
  AddVariables(final_norm_);
  AddVariable(&policy_linear_k_);
  AddVariable(&outcome_linear_k_);

  AddConstant(&input_);
  AddConstant(&policy_target_);
  AddConstant(&outcome_target_);
}

}  // namespace mancala
}  // namespace games
}  // namespace azah
