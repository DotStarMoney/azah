#ifndef AZAH_GAMES_TICTACTOE_TICTACTOE_H_
#define AZAH_GAMES_TICTACTOE_TICTACTOE_H_

#include <array>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "../../nn/data_types.h"
#include "../game.h"

namespace azah {
namespace games {
namespace tictactoe {

class Tictactoe : public Game<2, 9> {
 public:
  Tictactoe();
  const std::string_view name() const override;

  const std::string& state_uid() const override;

  int CurrentPlayerI() const override;
  int CurrentMovesN() const override;
  GameState State() const override;
  std::array<float, 2> Outcome() const override;

  std::vector<nn::DynamicMatrix> StateToMatrix() const override;
  int PolicyClassI() const override;
  float PolicyForMoveI(const std::span<float const>& policy,
                       int move_i) const override;
  
  nn::DynamicMatrix PolicyMask() const override;

  void MakeMove(int move_i) override;

 private:
  static constexpr std::string_view kName = "TicTacToe";

  void UpdateUid();

  enum class Mark {
    kNone = 0,
    kX = 1,
    kO = 2
  };

  std::array<Mark, 9> board_;
  bool x_move_;
  std::string uid_;
};

}  // namespace tictactoe
}  // namespace games
}  // namespace azah

#endif  // AZAH_GAMES_TICTACTOE_TICTACTOE_H_
