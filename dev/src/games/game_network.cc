#include "game_network.h"

#include <vector>

namespace azah {
namespace games {

GameNetwork::GameNetwork(
    std::vector<int>&& input_constant_indices,
    std::vector<int>&& policy_target_constant_indices,
    int outcome_target_constant_index,
    std::vector<int>&& policy_loss_target_indices,
    int outcome_loss_target_index,
    std::vector<int>&& policy_output_indices,
    int outcome_output_index,
    std::vector<int>&& policy_output_sizes) :
        input_constant_indices_(std::move(input_constant_indices)),
        policy_target_constant_indices_(std::move(policy_target_constant_indices)),
        outcome_target_constant_index_(outcome_target_constant_index),
        policy_loss_target_indices_(std::move(policy_loss_target_indices)),
        outcome_loss_target_index_(outcome_loss_target_index),
        policy_output_indices_(std::move(policy_output_indices)),
        outcome_output_index_(outcome_output_index),
        policy_output_sizes_(std::move(policy_output_sizes)) {}

const std::vector<int>& GameNetwork::InputConstantIndices() const {
  return input_constant_indices_;
}

const std::vector<int>& GameNetwork::PolicyTargetConstantIndices() const {
  return policy_target_constant_indices_;
}

int GameNetwork::OutcomeTargetConstantIndex() const {
  return outcome_target_constant_index_;
}

const std::vector<int>& GameNetwork::PolicyLossTargetIndices() const {
  return policy_loss_target_indices_;
}

int GameNetwork::OutcomeLossTargetIndex() const {
  return outcome_loss_target_index_;
}

const std::vector<int>& GameNetwork::PolicyOutputIndices() const {
  return policy_output_indices_;
}

int GameNetwork::OutcomeOutputIndex() const {
  return outcome_output_index_;
}

int GameNetwork::PolicyOutputSize(int policy_output_i) const {
  return policy_output_sizes_[policy_output_i];
}

}  // namespace games
}  // namespace azah
