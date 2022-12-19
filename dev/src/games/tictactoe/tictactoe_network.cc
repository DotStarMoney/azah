#include "tictactoe_network.h"

#include "../../nn/init.h"

namespace azah {
namespace games {
namespace tictactoe {

TictactoeNetwork::TictactoeNetwork() :
    input_(nn::init::Zeros<9, 1>()),
    dense1_k_(nn::init::GlorotUniform<kLayer1Depth, 9>()),
    dense1_(dense1_k_, input_),
    norm1_b_(nn::init::Zeros<kLayer1Depth, 1>()),
    norm1_g_(nn::init::Ones<kLayer1Depth, 1>()),
    norm1_(dense1_, norm1_b_, norm1_g_),
    swish1_(norm1_),
    dense2_k_(nn::init::GlorotUniform<kLayer2Depth, kLayer1Depth>()),
    dense2_(dense2_k_, swish1_),
    norm2_b_(nn::init::Zeros<kLayer2Depth, 1>()),
    norm2_g_(nn::init::Ones<kLayer2Depth, 1>()),
    norm2_(dense2_, norm2_b_, norm2_g_),
    swish2_(norm2_),
    policy_linear_k_(nn::init::GlorotUniform<9, kLayer2Depth>()),
    policy_linear_(policy_linear_k_, swish2_),
    policy_(policy_linear_),
    policy_target_(nn::init::Zeros<9, 1>()),
    policy_loss_(policy_linear_, policy_target_),
    outcome_linear_k_(nn::init::GlorotUniform<2, kLayer2Depth>()),
    outcome_linear_(outcome_linear_k_, swish2_),
    outcome_(outcome_linear_),
    outcome_target_(nn::init::Zeros<2, 1>()),
    outcome_loss_(outcome_linear_, outcome_target_) {
  AddOutput(&policy_);
  AddOutput(&outcome_);

  AddTarget(&policy_loss_);
  AddTarget(&outcome_loss_);

  AddVariable(&dense1_k_);
  AddVariable(&norm1_b_);
  AddVariable(&norm1_g_);
  AddVariable(&dense2_k_);
  AddVariable(&norm2_b_);
  AddVariable(&norm2_g_);
  AddVariable(&policy_linear_k_);
  AddVariable(&outcome_linear_k_);

  AddConstant(&input_, TictactoeNetwork::kInput);
  AddConstant(&policy_target_, TictactoeNetwork::kPolicyTarget);
  AddConstant(&outcome_target_, TictactoeNetwork::kOutcomeTarget);
}

}  // namespace tictactoe
}  // namespace games
}  // namespace azah
