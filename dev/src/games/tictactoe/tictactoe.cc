#include "tictactoe.h"

#include <vector>

#include "glog/logging.h"

namespace azah {
namespace games {
namespace tictactoe {

Tictactoe::Tictactoe() : 
    board_{Mark::kNone, Mark::kNone, Mark::kNone,
            Mark::kNone, Mark::kNone, Mark::kNone,
            Mark::kNone, Mark::kNone, Mark::kNone},
    x_move_(true) {
  UpdateUid();
}

const std::string_view Tictactoe::name() const {
  return kName;
}

const int Tictactoe::players_n() const {
  return 2;
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
    LOG(FATAL) << "Game is over, there is no current move.";
  }
  int total_moves = 0;
  for (int i = 0; i < 9; ++i) {
    if (board_[i] == Mark::kNone) ++total_moves;
  }
  return total_moves;
}

Game::GameState Tictactoe::State() const {
  for (int row = 0; row < 3; ++row) {
    if ((board_[row * 3 + 0] != Mark::kNone) 
        && (board_[row * 3 + 0] == board_[row * 3 + 1])
        && (board_[row * 3 + 1] == board_[row * 3 + 2])) {
      return GameState::kWinner;
    }
  }

  for (int col = 0; col < 3; ++col) {
    if ((board_[col] != Mark::kNone) 
        && (board_[col + 0] == board_[col + 3])
        && (board_[col + 3] == board_[col + 6])) {
      return GameState::kWinner;
    }
  }
  
  if ((board_[0] != Mark::kNone)
      && (board_[0] == board_[4])
      && (board_[4] == board_[8])) {
    return GameState::kWinner;
  }
  
  if ((board_[2] != Mark::kNone)
      && (board_[2] == board_[4])
      && (board_[4] == board_[6])) {
    return GameState::kWinner;
  }

  for (int i = 0; i < 9; ++i) {
    if (board_[i] == Mark::kNone) return GameState::kOngoing;
  }

  return GameState::kTie;
}

int Tictactoe::WinningPlayerI() const {
  if (State() != GameState::kWinner) {
    LOG(FATAL) << "The game is still going or it was a tie.";
  }

  for (int row = 0; row < 3; ++row) {
    if ((board_[row * 3 + 0] != Mark::kNone)
      && (board_[row * 3 + 0] == board_[row * 3 + 1])
      && (board_[row * 3 + 1] == board_[row * 3 + 2])) {
      return (board_[row * 3 + 0] == Mark::kX) ? 0 : 1;
    }
  }

  for (int col = 0; col < 3; ++col) {
    if ((board_[col] != Mark::kNone)
      && (board_[col + 0] == board_[col + 3])
      && (board_[col + 3] == board_[col + 6])) {
      return (board_[col] == Mark::kX) ? 0 : 1;
    }
  }

  if ((board_[0] != Mark::kNone)
    && (board_[0] == board_[4])
    && (board_[4] == board_[8])) {
    return (board_[0] == Mark::kX) ? 0 : 1;
  }

  if ((board_[2] != Mark::kNone)
    && (board_[2] == board_[4])
    && (board_[4] == board_[6])) {
    return (board_[2] == Mark::kX) ? 0 : 1;
  }

  LOG(FATAL) << "Assert... nobody won!";
}

int Tictactoe::PolicyToMoveI(const std::span<float>& policy) const {
  int max_index = -1;
  float max_value = 0;

  int move_index = 0;
  for (int i = 1; i < 9; ++i) {
    if (board_[i] != Mark::kNone) continue;
    if (policy[i] > max_value) {
      max_index = move_index;
      max_value = policy[i];
    }
    ++move_index;
  }

  if (max_index == -1) {
    LOG(FATAL) << "Assert... there was no move to make!";
  }

  return max_index;
}

void Tictactoe::MakeMove(int move_i) {
  if (State() != GameState::kOngoing) {
    LOG(FATAL) << "Game is over, there are no moves to make.";
  }

  int move_index = 0;
  for (int i = 1; i < 9; ++i) {
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
  char repr[10];
  for (int i = 0; i < 9; ++i) {
    repr[i] = static_cast<char>(board_[i]);
  }
  repr[9] = '\0';
  uid_ = repr;
}

}  // namespace tictactoe
}  // namespace games
}  // namespace azah
