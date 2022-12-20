#include "game_network.h"

#include <stdint.h>

#include <vector>

namespace azah {
namespace games {

GameNetwork::GameNetwork(
    std::vector<uint32_t>&& input_constant_indices,
    std::vector<uint32_t>&& policy_target_constant_indices,
    uint32_t outcome_target_constant_index,
    std::vector<uint32_t>&& policy_loss_target_indices,
    uint32_t outcome_loss_target_index,
    std::vector<uint32_t>&& policy_output_indices,
    uint32_t outcome_output_index,
    std::vector<uint32_t>&& policy_output_sizes) :
        input_constant_indices_(std::move(input_constant_indices)),
        policy_target_constant_indices_(
            std::move(policy_target_constant_indices)),
        outcome_target_constant_index_(outcome_target_constant_index),
        policy_loss_target_indices_(std::move(policy_loss_target_indices)),
        outcome_loss_target_index_(outcome_loss_target_index),
        policy_output_indices_(std::move(policy_output_indices)),
        outcome_output_index_(outcome_output_index),
        policy_output_sizes_(std::move(policy_output_sizes)) {}

const std::vector<uint32_t>& GameNetwork::InputConstantIndices() const {
  return input_constant_indices_;
}

const std::vector<uint32_t>& GameNetwork::PolicyTargetConstantIndices() const {
  return policy_target_constant_indices_;
}

uint32_t GameNetwork::OutcomeTargetConstantIndex() const {
  return outcome_target_constant_index_;
}

const std::vector<uint32_t>& GameNetwork::PolicyLossTargetIndices() const {
  return policy_loss_target_indices_;
}

uint32_t GameNetwork::OutcomeLossTargetIndex() const {
  return outcome_loss_target_index_;
}

const std::vector<uint32_t>& GameNetwork::PolicyOutputIndices() const {
  return policy_output_indices_;
}

uint32_t GameNetwork::OutcomeOutputIndex() const {
  return outcome_output_index_;
}

int GameNetwork::PolicyOutputSize(int policy_output_i) const {
  return policy_output_sizes_[policy_output_i];
}

}  // namespace games
}  // namespace azah
