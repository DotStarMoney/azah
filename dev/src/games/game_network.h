#ifndef AZAH_GAMES_GAME_NETWORK_H_
#define AZAH_GAMES_GAME_NETWORK_H_

#include <stdint.h>

#include <vector>

#include "../nn/network.h"

namespace azah {
namespace games {

class GameNetwork : public nn::Network {
 public:
  GameNetwork(const GameNetwork&) = delete;
  GameNetwork& operator=(const GameNetwork&) = delete;

  GameNetwork(
      std::vector<uint32_t>&& input_constant_indices,
      std::vector<uint32_t>&& policy_target_constant_indices,
      uint32_t outcome_target_constant_index,
      std::vector<uint32_t>&& policy_loss_target_indices,
      uint32_t outcome_loss_target_index,
      std::vector<uint32_t>&& policy_output_indices,
      uint32_t outcome_output_index,
      std::vector<uint32_t>&& policy_output_sizes);

  const std::vector<uint32_t>& input_constant_indices() const;
  
  const std::vector<uint32_t>& policy_target_constant_indices() const;
  uint32_t outcome_target_constant_index() const;
  
  const std::vector<uint32_t>& policy_loss_target_indices() const;
  uint32_t outcome_loss_target_index() const;

  const std::vector<uint32_t>& policy_output_indices() const;
  uint32_t outcome_output_index() const;

  int policy_output_sizes(int policy_output_i) const;

 private:
  const std::vector<uint32_t> input_constant_indices_;

  const std::vector<uint32_t> policy_target_constant_indices_;
  const uint32_t outcome_target_constant_index_;

  const std::vector<uint32_t> policy_loss_target_indices_;
  const uint32_t outcome_loss_target_index_;

  const std::vector<uint32_t> policy_output_indices_;
  const uint32_t outcome_output_index_;

  const std::vector<uint32_t> policy_output_sizes_;
};

}  // namespace games
}  // namespace azah

#endif  // AZAH_GAMES_GAME_NETWORK_H_
