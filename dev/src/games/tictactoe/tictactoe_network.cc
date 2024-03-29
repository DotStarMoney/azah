#include "tictactoe_network.h"

#include "../../nn/init.h"
#include "../game_network.h"

namespace azah {
namespace games {
namespace tictactoe {

TictactoeNetwork::TictactoeNetwork() :
    GameNetwork({0}, {1}, 2, {0}, 1, {0}, 1),
    input_(nn::init::Zeros<9, 1>()),
    dense1_k_(nn::init::GlorotUniform<kLayer1Depth, 9>()),
    dense1_(dense1_k_, input_),
    norm1_(dense1_),
    swish1_(norm1_),
    dense2_k_(nn::init::GlorotUniform<kLayer2Depth, kLayer1Depth>()),
    dense2_(dense2_k_, swish1_),
    norm2_(dense2_),
    swish2_(norm2_),
    swish2_fork_(swish2_, 2),
    policy_linear_k_(nn::init::GlorotUniform<9, kLayer2Depth>()),
    policy_linear_(policy_linear_k_, swish2_fork_),
    policy_(policy_linear_),
    policy_target_(nn::init::Zeros<9, 1>()),
    policy_loss_(policy_linear_, policy_target_),
    outcome_linear_k_(nn::init::GlorotUniform<2, kLayer2Depth>()),
    outcome_linear_(outcome_linear_k_, swish2_fork_),
    outcome_(outcome_linear_),
    outcome_target_(nn::init::Zeros<2, 1>()),
    outcome_loss_(outcome_linear_, outcome_target_) {
  AddOutput(&policy_);
  AddOutput(&outcome_);

  AddTarget(&policy_loss_);
  AddTarget(&outcome_loss_);

  AddVariable(&dense1_k_);
  AddVariables(norm1_);
  AddVariable(&dense2_k_);
  AddVariables(norm2_);
  AddVariable(&policy_linear_k_);
  AddVariable(&outcome_linear_k_);

  AddConstant(&input_);
  AddConstant(&policy_target_);
  AddConstant(&outcome_target_);
}

}  // namespace tictactoe
}  // namespace games
}  // namespace azah
