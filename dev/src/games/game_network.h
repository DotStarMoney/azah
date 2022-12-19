#ifndef AZAH_GAMES_GAME_NETWORK_H_
#define AZAH_GAMES_GAME_NETWORK_H_

#include <vector>

#include "../nn/network.h"

namespace azah {
namespace games {

class GameNetwork : public nn::Network {
 public:
  GameNetwork(const GameNetwork&) = delete;
  GameNetwork& operator=(const GameNetwork&) = delete;

  GameNetwork(
      std::vector<int>&& input_constant_indices,
      std::vector<int>&& policy_target_constant_indices,
      int outcome_target_constant_index,
      std::vector<int>&& policy_loss_target_indices,
      int outcome_loss_target_index,
      std::vector<int>&& policy_output_indices,
      int outcome_output_index,
      std::vector<int>&& policy_output_sizes);

  const std::vector<int>& InputConstantIndices() const;
  
  const std::vector<int>& PolicyTargetConstantIndices() const;
  int OutcomeTargetConstantIndex() const;
  
  const std::vector<int>& PolicyLossTargetIndices() const;
  int OutcomeLossTargetIndex() const;

  const std::vector<int>& PolicyOutputIndices() const;
  int OutcomeOutputIndex() const;

  int PolicyOutputSize(int policy_output_i) const;

 private:
  const std::vector<int> input_constant_indices_;

  const std::vector<int> policy_target_constant_indices_;
  const int outcome_target_constant_index_;

  const std::vector<int> policy_loss_target_indices_;
  const int outcome_loss_target_index_;

  const std::vector<int> policy_output_indices_;
  const int outcome_output_index_;

  const std::vector<int> policy_output_sizes_;
};

}  // namespace games
}  // namespace azah

#endif  // AZAH_GAMES_GAME_NETWORK_H_
