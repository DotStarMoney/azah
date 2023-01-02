#ifndef AZAH_GAMES_MANCALA_MANCALA_H_
#define AZAH_GAMES_MANCALA_MANCALA_H_

#include <stddef.h>

#include <array>
#include <string>
#include <string_view>
#include <vector>

#include "../../nn/data_types.h"
#include "../game.h"

namespace azah {
namespace games {
namespace mancala {

class Mancala : public Game<2> {
 public:
  Mancala();
  const std::string_view name() const override;

  int CurrentPlayerI() const override;
  int CurrentMovesN() const override;
  GameState State() const override;
  std::array<float, 2> Outcome() const override;

  std::vector<nn::DynamicMatrix> StateToMatrix() const override;
  int PolicyClassI() const override;
  float PolicyForMoveI(const nn::DynamicMatrix& policy,
                       int move_i) const override;

  nn::DynamicMatrix PolicyMask() const override;

  void MakeMove(int move_i) override;

 private:
  static constexpr std::string_view kName_ = "Mancala";

  static constexpr std::size_t kPlayerAWellI_ = 6;
  static constexpr std::size_t kPlayerBWellI_ = 13;

  bool player_a_turn_;
  std::array<int, 14> board_;
  bool over_;
  
  // This tracks the filled pockets for the current player, so won't ever exceed
  // 6 in length and no element will be > 5.
  std::array<std::size_t, 6> filled_pockets_;
  std::size_t filled_pockets_n_;
};

}  // namespace mancala
}  // namespace games
}  // namespace azah

#endif  // AZAH_GAMES_MANCALA_MANCALA_H_
