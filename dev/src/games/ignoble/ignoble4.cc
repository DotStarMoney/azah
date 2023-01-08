#include "ignoble4.h"

#include <stddef.h>

#include <algorithm>
#include <array>
#include <string_view>
#include <vector>

#include "../../nn/data_types.h"
#include "../../nn/init.h"
#include "../game.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"

namespace azah {
namespace games {
namespace ignoble {
namespace {

constexpr int kSoilOffset = 0;
constexpr int kHerbOffset = 23;
constexpr int kBeastOffset = 46;
constexpr int kCoinOffset = 69;

}  // namespace

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
  std::vector<nn::DynamicMatrix> inputs;
  for (std::size_t i = 0; i < 4; ++i) {
    inputs.push_back(nn::init::Zeros<128, 1>());
    auto& f = inputs.back();
    std::size_t player_i = (current_player_i_ + i) % 4;
    
    // First the stock amounts, which are length 23 segments. The first 22 elements
    // are a one-hot vector of the discrete amount of that stock owned, and the
    // 23rd marks that the stock is "overflowing" the required amount.

    int soil = soil_n_[player_i];
    f(kSoilOffset + soil, 0) = 1.0f;
    if (soil > 5) f(kSoilOffset + 22) = 1.0f;

    int herb = herb_n_[player_i];
    f(kHerbOffset + herb, 0) = 1.0f;
    if (herb > 5) f(kHerbOffset + 22) = 1.0f;

    int beast = beast_n_[player_i];
    f(kBeastOffset + beast, 0) = 1.0f;
    if (beast > 5) f(kBeastOffset + 22) = 1.0f;

    int coin = coin_n_[player_i];
    f(kCoinOffset + coin, 0) = 1.0f;
    if (coin > 5) f(kCoinOffset + 22) = 1.0f;



  }
  return inputs;
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
