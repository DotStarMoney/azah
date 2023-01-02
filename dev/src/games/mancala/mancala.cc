#include "mancala.h"

#include <array>
#include <string>
#include <string_view>
#include <vector>

#include "../../nn/data_types.h"
#include "../../nn/init.h"
#include "../game.h"

namespace azah {
namespace games {
namespace mancala {

Mancala::Mancala() : 
    player_a_turn_(true),
    board_{4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0},
    over_(false) {
  
}

const std::string_view Mancala::name() const {
  return kName_;
}

int Mancala::CurrentPlayerI() const {
  return player_a_turn_ ? 0 : 1;
}

int Mancala::CurrentMovesN() const {

}

GameState Mancala::State() const {
  return over_ ? GameState::kOver : GameState::kOngoing;
}

std::array<float, 2> Mancala::Outcome() const {
  return (board_[kPlayerAWellI_] > board_[kPlayerBWellI_])
      ? std::array<float, 2>{1.0f, 0.0f}
      : std::array<float, 2>{0.0f, 1.0f};
}

std::vector<nn::DynamicMatrix> Mancala::StateToMatrix() const {
  nn::Matrix<14, 48> input_ = nn::init::Zeros<14, 48>();
  for (std::size_t i = 0; i < 14; ++i) {
    input_(i, board_[i]) = 1.0f;
  }
  return {input_};
}

int Mancala::PolicyClassI() const {
  return 0;
}

float Mancala::PolicyForMoveI(const nn::DynamicMatrix& policy,
                              int move_i) const {

}

nn::DynamicMatrix Mancala::PolicyMask() const {

}

void Mancala::MakeMove(int move_i) {

  player_a_turn_ = !player_a_turn_;
}

}  // namespace mancala
}  // namespace games
}  // namespace azah
