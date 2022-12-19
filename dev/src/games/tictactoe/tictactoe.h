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
  const std::string_view name() const;

  const std::string& state_uid() const;

  int inputs_model_tag() const;
  int target_policies_model_tag() const;
  int target_outcomes_model_tag() const;

  int CurrentPlayerI() const;
  int CurrentMovesN() const;
  GameState State() const;
  std::array<float, 2> Outcome() const;

  std::vector<nn::DynamicMatrix> StateToMatrix() const;
  int PolicyToMoveI(std::span<float const> policy) const;
  int PolicyClassI() const;
  nn::DynamicMatrix MoveVisitCountToPolicy(std::span<int const> visits) const;

  void MakeMove(int move_i);

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
