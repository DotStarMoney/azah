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
#include "glog/logging.h"

#undef max

namespace azah {
namespace games {
namespace ignoble {
namespace {

// The stock state offsets in order of soil, herb, beast, coin.
constexpr int kStockStateOffsets[] = {0, 23, 46, 69};

constexpr int kHandCardsOffset = 92;
constexpr int kPlayedCardsOffset = 108;
constexpr int kPlayerCardOffset = 124;

constexpr int kTieOrderOffset = 140;

constexpr int kLocationOffset = 0;
constexpr int kRemainingLocations = 48;

// Decisions where cards are in play.
constexpr bool kCardsInPlayDecisions[] = {
        false,  // Unknown
        false,  // Selecting team 
        true,   // Selecting character
        true,   // Princess
        true,   // Meat Bungler
        true,   // Merry Pieman
        true,   // Benedict
        true,   // Bethesda
        true,   // Ounce
        true,   // Magician
        false   // Repent
    };

constexpr int kMaxDecisionsN[] = {
        4,   // Selecting team 
        4,   // Selecting character
        4,   // Princess
        3,   // Meat Bungler
        4,   // Merry Pieman
        2,   // Benedict
        2,   // Bethesda
        12,  // Ounce
        8,   // Magician
        4    // Repent
    };

struct Location {
  // 0=soil, 1=herb, 2=beast, 3=coin
  int type;
  int bounty_n;
};
constexpr Location kLocations[] = {
        {0, 2},  // Soot Cellar
        {0, 3},  // Swamp
        {0, 3},  // Peasant Village
        {1, 1},  // The Hedgerow
        {1, 2},  // Grove
        {1, 3},  // Vegetable Garden
        {2, 1},  // Thy Frigid Plunge
        {2, 2},  // Pig Sty
        {2, 2},  // Plains
        {3, 1},  // Market
        {3, 1},  // The Tomb
        {3, 2}   // The Coiniary
    };

}  // namespace

Ignoble4::Ignoble4() :
    decision_class_(Decisions::kTeamSelect),
    deck_select_tie_order_{0, 1, 2, 3},
    hand_size_{0, 0, 0, 0},
    decks_available_{true, true, true, true},
    player_turn_i_(0),
    stock_n_{{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
    current_location_i_(0),
    stock_modifier(0),
    location_deck_{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
    top_of_deck_i_(7),
    winning_player_i_(-1),
    available_actions_n_(4),
    move_to_policy_i_{0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} {
  absl::BitGen bitgen;
  // This will never change, and is the equivalent of picking a player to start
  // and the seating order at the table.
  std::shuffle(deck_select_tie_order_.begin(), deck_select_tie_order_.end(), 
               bitgen);
  std::shuffle(location_deck_.begin(), location_deck_.end(), bitgen);
  locations_in_play_[0] = location_deck_[11];
  locations_in_play_[1] = location_deck_[10];
  locations_in_play_[2] = location_deck_[9];
  locations_in_play_[3] = location_deck_[8];
  current_player_i_ = deck_select_tie_order_[0];
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
  // This provides 5 inputs:
  //   - 4x player state as a 144-D vector.
  //   - 1x global state as a 60-D vector.
  // 
  // The player currently making a decision is rotated into the first slot.
  std::vector<nn::DynamicMatrix> inputs;
  for (std::size_t i = 0; i < 4; ++i) {
    inputs.push_back(nn::init::Zeros<144, 1>());
    auto& f = inputs.back();
    std::size_t player_i = (current_player_i_ + i) % 4;
    
    // First the stock amounts, which are length 23 segments. The first 22 
    // elements are a one-hot vector of the discrete amount of that stock owned,
    // and the 23rd marks that the stock is "overflowing" the required amount.

    for (int stock_i = 0; stock_i < 4; ++stock_i) {
      int stock = stock_n_[player_i][stock_i];
      f(kStockStateOffsets[stock_i] + stock, 0) = 1.0f;
      if (stock > 5) f(kStockStateOffsets[stock_i] + 22) = 1.0f;
    }

    // 16 Multi-hot cards still in the player's hand.
    
    for (std::size_t card_i = 0; card_i < hand_size_[player_i]; ++card_i) {
      f(kHandCardsOffset + hand_[player_i][card_i], 0) = 1.0f;
    }

    if (kCardsInPlayDecisions[PolicyClassI()]) {
      // 16 Multi-hot cards in play.
      for (std::size_t card_i = 0; card_i < 4; ++card_i) {
        f(kPlayedCardsOffset + cards_in_play_[card_i].value, 0) = 1.0f;
        // The card played by player_i.
        if (cards_in_play_[card_i].player_i == player_i) {
          f(kPlayerCardOffset + cards_in_play_[card_i].value, 0) = 1.0f;
        }
      }
    }

    f(kTieOrderOffset + deck_select_tie_order_[player_i]) = 1.0f;
  }

  // Now the final input which is the global board state (so locations).
  inputs.push_back(nn::init::Zeros<60, 1>());
  auto& g = inputs.back();

  // The locations out at t0, t1, etc...
  for (std::size_t location_i = current_location_i_, slot_i = 0; 
       location_i < 4;
       ++location_i, ++slot_i) {
    g(kLocationOffset + slot_i * 12 + locations_in_play_[location_i], 0) = 1.0f;
  }

  // The locations not yet played.
  for (std::size_t card_i = 0; card_i <= top_of_deck_i_; ++card_i) {
    g(kRemainingLocations + location_deck_[card_i], 0) = 1.0f;
  }
 
  return inputs;
}

int Ignoble4::PolicyClassI() const {
  return static_cast<int>(decision_class_) - 1;
}

int Ignoble4::current_bounty() const {
  return std::max(
      kLocations[locations_in_play_[current_location_i_]].bounty_n 
          + stock_modifier, 
      0);
}

float Ignoble4::PolicyForMoveI(const nn::DynamicMatrix& policy,
                               int move_i) const {
  return policy(move_to_policy_i_[move_i], 0);
}

nn::DynamicMatrix Ignoble4::PolicyMask() const {
  nn::DynamicMatrix mask(kMaxDecisionsN[static_cast<int>(decision_class_)], 1);
  mask.setZero();
  for (std::size_t i = 0; i < available_actions_n_; ++i) mask(i, 0) = 1.0f;
  return mask;
}

void Ignoble4::MakeMove(int move_i, absl::BitGenRef bitgen) {
  //
}

}  // namespace ignoble
}  // namespace games
}  // namespace azah
