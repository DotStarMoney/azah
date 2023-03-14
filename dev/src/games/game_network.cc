#include "game_network.h"

#include <stdint.h>

#include <iostream>
#include <vector>

#include "../nn/data_types.h"

namespace azah {
namespace games {

GameNetwork::GameNetwork(
    std::vector<uint32_t>&& input_constant_indices,
    std::vector<uint32_t>&& policy_target_constant_indices,
    uint32_t outcome_target_constant_index,
    std::vector<uint32_t>&& policy_loss_target_indices,
    uint32_t outcome_loss_target_index,
    std::vector<uint32_t>&& policy_output_indices,
    uint32_t outcome_output_index) :
        input_constant_indices_(std::move(input_constant_indices)),
        policy_target_constant_indices_(
            std::move(policy_target_constant_indices)),
        outcome_target_constant_index_(outcome_target_constant_index),
        policy_loss_target_indices_(std::move(policy_loss_target_indices)),
        outcome_loss_target_index_(outcome_loss_target_index),
        policy_output_indices_(std::move(policy_output_indices)),
        outcome_output_index_(outcome_output_index) {}

const std::vector<uint32_t>& GameNetwork::input_constant_indices() const {
  return input_constant_indices_;
}

const std::vector<uint32_t>& 
    GameNetwork::policy_target_constant_indices() const {
  return policy_target_constant_indices_;
}

uint32_t GameNetwork::outcome_target_constant_index() const {
  return outcome_target_constant_index_;
}

const std::vector<uint32_t>& GameNetwork::policy_loss_target_indices() const {
  return policy_loss_target_indices_;
}

uint32_t GameNetwork::outcome_loss_target_index() const {
  return outcome_loss_target_index_;
}

const std::vector<uint32_t>& GameNetwork::policy_output_indices() const {
  return policy_output_indices_;
}

uint32_t GameNetwork::outcome_output_index() const {
  return outcome_output_index_;
}

void GameNetwork::Serialize(std::ostream& out) const {
  std::vector<nn::ConstDynamicMatrixRef> vars;
  GetVariables({}, vars);
  for (const auto& var : vars) {
    out.write(reinterpret_cast<const char*>(var.data()), 
              sizeof(float) * var.size());
  }
}

void GameNetwork::Deserialize(std::istream& in) {
  std::vector<nn::DynamicMatrixRef> vars;
  GetVariables({}, vars);
  for (auto& var : vars) {
    in.read(reinterpret_cast<char*>(var.data()), var.size());
  }
}

}  // namespace games
}  // namespace azah
