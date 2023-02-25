#ifndef AZAH_GAMES_TICTACTOE_TICTACTOE_H_
#define AZAH_GAMES_TICTACTOE_TICTACTOE_H_

#include <array>
#include <string_view>
#include <vector>

#include "../../nn/data_types.h"
#include "../game.h"

namespace azah {
namespace games {
namespace tictactoe {

class Tictactoe : public Game<2> {
 public:
  Tictactoe();
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
  static constexpr std::string_view kName_ = "TicTacToe";

  enum class Mark {
    kNone = 0,
    kX = 1,
    kO = 2
  };

  std::array<Mark, 9> board_;
  bool x_move_;
};

}  // namespace tictactoe
}  // namespace games
}  // namespace azah

#endif  // AZAH_GAMES_TICTACTOE_TICTACTOE_H_
