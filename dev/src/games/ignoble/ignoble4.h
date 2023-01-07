#ifndef AZAH_GAMES_IGNOBLE_IGNOBLE4_H_
#define AZAH_GAMES_IGNOBLE_IGNOBLE4_H_

#include <stddef.h>

#include <array>
#include <string_view>
#include <vector>

#include "../../nn/data_types.h"
#include "../game.h"
#include "absl/random/bit_gen_ref.h"

namespace azah {
namespace games {
namespace ignoble {

class Ignoble4 : public Game<4> {
public:
  Ignoble4();
  const std::string_view name() const override;

  int CurrentPlayerI() const override;
  int CurrentMovesN() const override;
  GameState State() const override;
  std::array<float, 4> Outcome() const override;

  std::vector<nn::DynamicMatrix> StateToMatrix() const override;
  int PolicyClassI() const override;
  float PolicyForMoveI(const nn::DynamicMatrix& policy, 
                       int move_i) const override;

  nn::DynamicMatrix PolicyMask() const override;

  void MakeMove(int move_i, absl::BitGenRef bitgen);

private:
  static constexpr std::string_view kName_ = "Ignoble 4-Player";

  enum class RoundPhase {
    kUnknown = 0,
    kTeamSelect = 1,
    kCharacterSelect = 2,
    kResolveActions = 3,
    kRepent = 4
  };
  RoundPhase round_phase_;


 
};

}  // namespace ignoble
}  // namespace games
}  // namespace azah

#endif  // AZAH_GAMES_IGNOBLE_IGNOBLE4_H_
