#include "ignoble4.h"

#include <algorithm>
#include <array>
#include <string_view>
#include <vector>

#include "../../nn/data_types.h"
#include "../game.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"

namespace azah {
namespace games {
namespace ignoble {

Ignoble4::Ignoble4() :
    round_phase_(RoundPhase::kTeamSelect),
    deck_select_tie_order_{0, 1, 2, 3},
    player_turn_i_(0),
    soil_n_{0, 0, 0, 0},
    herb_n_{0, 0, 0, 0},
    beast_n_{0, 0, 0, 0},
    coin_n_{0, 0, 0, 0},
    current_location_i_(0),
    location_deck_{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
    top_of_deck_i_(11),
    winning_player_i_(-1) {
  absl::BitGen bitgen;
  // This will never change, and is the equivalent of picking a player to start and
  // the seating order at the table.
  std::shuffle(deck_select_tie_order_.begin(), deck_select_tie_order_.end(), 
               bitgen);
  std::shuffle(location_deck_.begin(), location_deck_.end(), bitgen);
  current_player_i_ = deck_select_tie_order_[0];
  available_actions_n_ = 4;
  policy_class_i_ = 0;
}

const std::string_view Ignoble4::name() const {
  return kName_;
}

int Ignoble4::CurrentPlayerI() const {
  return current_player_i_;
}

int Ignoble4::CurrentMovesN() const {
  return available_actions_n_;
}

GameState Ignoble4::State() const {
  return (winning_player_i_ == -1) ? GameState::kOngoing : GameState::kOver;
}

std::array<float, 4> Ignoble4::Outcome() const {
  std::array<float, 4> outcome{0.0f, 0.0f, 0.0f, 0.0f};
  outcome[winning_player_i_] = 1.0f;
  return outcome;
}

std::vector<nn::DynamicMatrix> Ignoble4::StateToMatrix() const {
  //
  //
  //
  return {};
}

int Ignoble4::PolicyClassI() const {
  return policy_class_i_;
}

float Ignoble4::PolicyForMoveI(const nn::DynamicMatrix& policy,
                               int move_i) const {
  return 0.0f;
}

nn::DynamicMatrix Ignoble4::PolicyMask() const {
  return nn::DynamicMatrix();
}

void Ignoble4::MakeMove(int move_i, absl::BitGenRef bitgen) {
  //
}

}  // namespace ignoble
}  // namespace games
}  // namespace azah
