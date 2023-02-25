#include "mancala.h"

#include <array>
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
    over_(false),
    filled_pockets_{0, 1, 2, 3, 4, 5},
    filled_pockets_n_(6) {}

const std::string_view Mancala::name() const {
  return kName_;
}

int Mancala::CurrentPlayerI() const {
  return player_a_turn_ ? 0 : 1;
}

int Mancala::CurrentMovesN() const {
  return filled_pockets_n_;
}

GameState Mancala::State() const {
  return over_ ? GameState::kOver : GameState::kOngoing;
}

std::array<float, 2> Mancala::Outcome() const {
  if (board_[kPlayerAWellI_] > board_[kPlayerBWellI_]) {
    return std::array<float, 2>{1.0f, 0.0f};
  } else if (board_[kPlayerAWellI_] < board_[kPlayerBWellI_]) {
    return std::array<float, 2>{0.0f, 1.0f};
  } else {
    return std::array<float, 2>{0.5f, 0.5f};
  }
}

std::vector<nn::DynamicMatrix> Mancala::StateToMatrix() const {
  // 14 column vectors of height 48, where each vector is a separate board
  // space.
  nn::Matrix<48, 14> input = nn::init::Zeros<48, 14>();
  // If it's player B's turn, we rotate the board so that the first 7 columns
  // in the output matrix belong to B not A.
  if (player_a_turn_) {
    for (std::size_t i = 0; i < 14; ++i) {
      input(board_[i], i) = 1.0f;
    }
  } else {
    for (std::size_t i = 0; i < 14; ++i) {
      input(board_[(7 + i) % 14], i) = 1.0f;
    }
  }
  return {input};
}

int Mancala::PolicyClassI() const {
  return 0;
}

float Mancala::PolicyForMoveI(const nn::DynamicMatrix& policy,
                              int move_i) const {
  return policy(filled_pockets_[move_i], 0);
}

nn::DynamicMatrix Mancala::PolicyMask() const {
  nn::Matrix<6, 1> mask = nn::init::Zeros<6, 1>();
  for (std::size_t i = 0; i < filled_pockets_n_; ++i) {
    mask(filled_pockets_[i], 0) = 1.0f;
  }
  return mask;
}

void Mancala::MakeMove(int move_i) {
  // Since filled pockets is local to the current player, adjust it to a board
  // position.
  std::size_t well_index = filled_pockets_[move_i] + (player_a_turn_ ? 0 : 7);
  int pocket_contents = board_[well_index];
  board_[well_index] = 0;

  std::size_t opposing_mancala = 
      player_a_turn_ ? kPlayerBWellI_ : kPlayerAWellI_;
  std::size_t last_sow_space = -1;
  // Sow seeds, skipping the opponent's mancala and tracking the last place we 
  // left a stone.
  for (int i = 1; i <= pocket_contents; ++i) {
    std::size_t sow_space = (well_index + i) % 14;
    if (sow_space == opposing_mancala) {
      ++pocket_contents;
      continue;
    }
    ++(board_[sow_space]);
    last_sow_space = sow_space;
  }

  // Get a LB / UB for our board side.
  auto [lb, ub] = player_a_turn_
      ? std::make_tuple<>(0, 6)
      : std::make_tuple<>(7, 13);

  // If our last stone was on our side, steal from our opponent and take the
  // stone.
  std::size_t mancala = player_a_turn_ ? kPlayerAWellI_ : kPlayerBWellI_;
  if ((board_[last_sow_space] == 1) &&
      ((last_sow_space >= lb) && (last_sow_space < ub))) {
    std::size_t opposing_pocket = ((ub - last_sow_space + lb) + 6) % 14;
    board_[mancala] += board_[opposing_pocket] + 1;
    board_[opposing_pocket] = 0;
    board_[last_sow_space] = 0;
  }

  std::size_t side_a_stones = 0;
  std::size_t side_b_stones = 0;
  for (int i = 0; i < 6; ++i) {
    side_a_stones += board_[i];
    side_b_stones += board_[7 + i];
  }
  over_ = (side_a_stones == 0) || (side_b_stones == 0);

  if (over_) {
    board_[kPlayerAWellI_] += side_a_stones;
    board_[kPlayerBWellI_] += side_b_stones;
    // Clearing here isn't neccessary, but is nice to do anyway.
    for (int i = 0; i < 6; ++i) {
      board_[i] = 0;
      board_[7 + i] = 0;
    }
    return;
  }

  if (last_sow_space != mancala) {
    // If we didn't place a stone in our own mancala switch the turn to be of
    // our opponent, and change the LB / UB to their side.
    player_a_turn_ = !player_a_turn_;
    lb = (lb + 7) % 14;
    ub = (ub + 7) % 14;
  }
  // Note that if we placed our last stone in our own mancala, its still our
  // turn (we never switched turns in the above if clause).

  // Now calculate available moves for our opponent (or us if we get another
  // turn).
  filled_pockets_n_ = 0;
  for (int i = lb; i < ub; ++i) {
    if (board_[i] == 0) continue;
    filled_pockets_[filled_pockets_n_++] = i - lb;
  }
}

}  // namespace mancala
}  // namespace games
}  // namespace azah
