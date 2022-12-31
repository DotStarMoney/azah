#include "tictactoe.h"

#include <array>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "../../nn/data_types.h"
#include "../../nn/init.h"
#include "glog/logging.h"

namespace azah {
namespace games {
namespace tictactoe {

Tictactoe::Tictactoe() : 
    board_{Mark::kNone, Mark::kNone, Mark::kNone,
           Mark::kNone, Mark::kNone, Mark::kNone,
           Mark::kNone, Mark::kNone, Mark::kNone},
    x_move_(true),
    uid_("_________") {
  UpdateUid();
}

const std::string_view Tictactoe::name() const {
  return kName;
}

const std::string& Tictactoe::state_uid() const {
  return uid_;
}

int Tictactoe::CurrentPlayerI() const {
  if (State() != GameState::kOngoing) {
    LOG(FATAL) << "Game is over, there is no current player.";
  }
  return x_move_ ? 0 : 1;
}

int Tictactoe::CurrentMovesN() const {
  if (State() != GameState::kOngoing) {
    LOG(FATAL) << "Game is over, there is no current move. Move is " 
        << static_cast<int>(State());
  }
  int total_moves = 0;
  for (int i = 0; i < 9; ++i) {
    if (board_[i] == Mark::kNone) ++total_moves;
  }
  return total_moves;
}

GameState Tictactoe::State() const {
  for (int row = 0; row < 3; ++row) {
    if ((board_[row * 3 + 0] != Mark::kNone)
        && (board_[row * 3 + 0] == board_[row * 3 + 1])
        && (board_[row * 3 + 1] == board_[row * 3 + 2])) {
      return GameState::kOver;
    }
  }

  for (int col = 0; col < 3; ++col) {
    if ((board_[col] != Mark::kNone)
        && (board_[col + 0] == board_[col + 3])
        && (board_[col + 3] == board_[col + 6])) {
      return GameState::kOver;
    }
  }

  if ((board_[0] != Mark::kNone)
      && (board_[0] == board_[4])
      && (board_[4] == board_[8])) {
    return GameState::kOver;
  }

  if ((board_[2] != Mark::kNone)
      && (board_[2] == board_[4])
      && (board_[4] == board_[6])) {
    return GameState::kOver;
  }

  for (int i = 0; i < 9; ++i) {
    if (board_[i] == Mark::kNone) return GameState::kOngoing;
  }

  return GameState::kOver;
}

std::array<float, 2> Tictactoe::Outcome() const {
  if (State() != GameState::kOver) {
    LOG(FATAL) << "The game is still going or it was a tie.";
  }

  for (int row = 0; row < 3; ++row) {
    if ((board_[row * 3 + 0] != Mark::kNone)
        && (board_[row * 3 + 0] == board_[row * 3 + 1])
        && (board_[row * 3 + 1] == board_[row * 3 + 2])) {
      return (board_[row * 3 + 0] == Mark::kX) 
          ? std::array{1.0f, 0.0f} 
          : std::array{0.0f, 1.0f};
    }
  }

  for (int col = 0; col < 3; ++col) {
    if ((board_[col] != Mark::kNone)
        && (board_[col + 0] == board_[col + 3])
        && (board_[col + 3] == board_[col + 6])) {
      return (board_[col] == Mark::kX) 
          ? std::array{1.0f, 0.0f}
          : std::array{0.0f, 1.0f};
    }
  }

  if ((board_[0] != Mark::kNone)
      && (board_[0] == board_[4])
      && (board_[4] == board_[8])) {
    return (board_[0] == Mark::kX)
        ? std::array{1.0f, 0.0f}
        : std::array{0.0f, 1.0f};
  }

  if ((board_[2] != Mark::kNone)
      && (board_[2] == board_[4])
      && (board_[4] == board_[6])) {
    return (board_[2] == Mark::kX)
        ? std::array{1.0f, 0.0f}
        : std::array{0.0f, 1.0f};
  }

  return std::array{0.5f, 0.5f};
}

std::vector<nn::DynamicMatrix> Tictactoe::StateToMatrix() const {
  nn::Matrix<9, 1> out;
  for (int i = 0; i < 9; ++i) {
    out(i, 0) =
        std::array{0.0f, 1.0f, -1.0f}[static_cast<int>(board_[i])];
  }
  if (!x_move_) out *= -1.0f;
  return {std::move(out)};
}

float Tictactoe::PolicyForMoveI(const nn::DynamicMatrix& policy,
                                int move_i) const {
  int move_index = 0;
  for (int i = 0; i < 9; ++i) {
    if (board_[i] != Mark::kNone) continue;
    if (move_index == move_i) return policy(i, 0);
    ++move_index;
  }

  LOG(FATAL) << "There is no move_i";
}

int Tictactoe::PolicyClassI() const {
  return 0;
}

nn::DynamicMatrix Tictactoe::PolicyMask() const {
  if (State() == GameState::kOver) {
    LOG(FATAL) << "Game is over, there is no policy.";
  }

  nn::Matrix<9, 1> policy = nn::init::Zeros<9, 1>();
  for (int i = 0; i < 9; ++i) {
    if (board_[i] != Mark::kNone) continue;
    policy(i, 0) = 1.0f;
  }

  return policy;
}

void Tictactoe::MakeMove(int move_i) {
  if (State() != GameState::kOngoing) {
    LOG(FATAL) << "Game is over, there are no moves to make.";
  }

  int move_index = 0;
  for (int i = 0; i < 9; ++i) {
    if (board_[i] != Mark::kNone) continue;
    if (move_index == move_i) {
      board_[i] = x_move_ ? Mark::kX : Mark::kO;
      x_move_ = !x_move_;
      UpdateUid();
      return;
    }
    ++move_index;
  }

  LOG(FATAL) << "The given move did not exist!";
}

void Tictactoe::UpdateUid() {
  for (int i = 0; i < 9; ++i) {
    uid_[i] = static_cast<char>(board_[i]);
  }
}

}  // namespace tictactoe
}  // namespace games
}  // namespace azah
