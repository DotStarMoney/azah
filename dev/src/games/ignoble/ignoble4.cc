#include "ignoble4.h"

#include <stddef.h>

#include <algorithm>
#include <array>
#include <string_view>
#include <vector>

#include "../../nn/data_types.h"
#include "../../nn/init.h"
#include "../coroutine.h"
#include "../game.h"
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
constexpr int kPlayerCardOffset = 108;
constexpr int kTieOrderOffset = 124;
constexpr int kOunceHotSeat = 128;

constexpr int kLocationOffset = 0;
constexpr int kRemainingLocations = 48;

constexpr int kMagicianIndex = 14;
constexpr int kKingIndex = 13;
constexpr int kOunceIndex = 12;
constexpr int kSirStrawberryIndex = 11;
constexpr int kBethesdaIndex = 10;
constexpr int kBenedictIndex = 9;
constexpr int kPiemanIndex = 8;
constexpr int kBunglerIndex = 7;
constexpr int kPinderIndex = 6;
constexpr int kPrincessIndex = 5;
constexpr int kBroomemanIndex = 4;
constexpr int kPageboyIndex = 3;
constexpr int kKnaveIndex = 2;
constexpr int kTavernFoolIndex = 1;

// Decisions where cards are in play.
constexpr bool kCardsInPlayDecisions[] = {
        false,  // Unknown
        false,  // Selecting team 
        false,  // Selecting character
        true,   // Princess
        true,   // Meat Bungler
        true,   // Merry Pieman
        true,   // Benedict
        true,   // Bethesda
        true,   // Ounce
        true,   // Magician
        false   // Repent
    };

// The size of the column vector representing an individual decision.
constexpr int kMaxDecisionRowsN[] = {
        4,   // Selecting team 
        16,  // Selecting character
        4,   // Princess
        3,   // Meat Bungler
        4,   // Merry Pieman
        2,   // Benedict
        16,  // Bethesda 
        4,   // Ounce
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

constexpr std::array<std::array<int, 4>, 4> kDeckContents = {{
        {2, 7, 9, 12},   // The Ounce
        {0, 6, 11, 13},  // The King
        {3, 5, 8, 14},   // The Magician
        {1, 4, 10, 15}   // Death
    }};

}  // namespace

Ignoble4::Ignoble4() :
    decision_class_(Decisions::kTeamSelect),
    deck_select_tie_order_{0, 1, 2, 3},
    hand_size_{0, 0, 0, 0},
    stock_n_{{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
    location_deck_{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
    top_of_deck_i_(-1),
    winning_player_i_(-1) {
  run_handle_ = RunGame();
}

Ignoble4::~Ignoble4() {
  run_handle_.destroy();
}

const std::string_view Ignoble4::name() const {
  return kName_;
}

int Ignoble4::CurrentPlayerI() const {
  return current_player_x_;
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
  //   - 4x player state as a 129-D vector.
  //   - 1x global state as a 60-D vector.
  // 
  // The player currently making a decision is rotated into the first slot.
  std::vector<nn::DynamicMatrix> inputs;
  for (std::size_t i = 0; i < 4; ++i) {
    inputs.push_back(nn::init::Zeros<144, 1>());
    auto& f = inputs.back();
    std::size_t player_i = (current_player_x_ + i) % 4;
    
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

    if (kCardsInPlayDecisions[static_cast<int>(decision_class_)]) {
      // 16 One-hot card in play.
      for (std::size_t card_i = 0; card_i < 4; ++card_i) {
        if (cards_in_play_[card_i].player_i == player_i) {
          f(kPlayerCardOffset + cards_in_play_[card_i].value, 0) = 1.0f;
          break;
        }
      }
    }

    f(kTieOrderOffset + deck_select_tie_order_[player_i]) = 1.0f;

    if ((decision_class_ == Decisions::kOunceStealStock) 
        && (player_i == ounce_hot_seat_)) {
      f(kOunceHotSeat) = 1.0f;
    }
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
  if (decision_class_ == Decisions::kUnknown) {
    LOG(FATAL) << "Unknown policy class.";
  }
  return static_cast<int>(decision_class_) - 1;
}

bool Ignoble4::ComparePlayerPickOrder(int player_a, int player_b) const {
  // Whoever has the least amount of the good stuff goes first.
  for (std::size_t stock_i = 0; stock_i < 4; ++stock_i) {
    if (stock_n_[player_a][3] < stock_n_[player_b][3]) {
      return true;
    } else if (stock_n_[player_a][3] > stock_n_[player_b][3]) {
      return false;
    }
  }
  // If we got here, both players have equivalent stock. Now we sort by tie
  // order.
  for (std::size_t tie_order_i = 0; tie_order_i < 4; ++tie_order_i) {
    if (deck_select_tie_order_[tie_order_i] == player_a) {
      return true;
    } else if (deck_select_tie_order_[tie_order_i] == player_b) {
      return false;
    }
  }
  LOG(FATAL) << "Failed to determine pick order between players " << player_a 
      << " and " << player_b << ".";
}

float Ignoble4::PolicyForMoveI(const nn::DynamicMatrix& policy,
                               int move_i) const {
  return policy(move_to_policy_i_[move_i], 0);
}

nn::DynamicMatrix Ignoble4::PolicyMask() const {
  nn::DynamicMatrix mask(
      kMaxDecisionRowsN[static_cast<int>(decision_class_)], 1);
  mask.setZero();
  for (std::size_t i = 0; i < available_actions_n_; ++i) {
    mask(move_to_policy_i_[i], 0) = 1.0f;
  }
  return mask;
}

void Ignoble4::MakeMove(int move_i) {
  move_i_ = move_i;
  run_handle_();
}

coroutine::Void Ignoble4::RunGame() {
  // This will never change, and is the equivalent of picking a player to start
  // and the seating order at the table.
  std::shuffle(deck_select_tie_order_.begin(), deck_select_tie_order_.end(),
               bitgen_);
  for (;;) {
    // Start of a new round. First we check to see if a location shuffle is in
    // order.
    if (top_of_deck_i_ == -1) {
      std::shuffle(location_deck_.begin(), location_deck_.end(), bitgen_);
      top_of_deck_i_ = 11;
    }

    // Deal the four locations.
    for (IndexT i = 0; i < 4; ++i) {
      locations_in_play_[i] = location_deck_[top_of_deck_i_ - i];
    }
    top_of_deck_i_ -= 4;

    // Figure out the pick order.
    std::array<IndexT, 4> pick_order{0, 1, 2, 3};
    for (IndexT i = 1; i < 4; ++i) {
      IndexT q = i - 1;
      while ((q >= 0) && ComparePlayerPickOrder(q + 1, q)) {
        std::swap(pick_order[q + 1], pick_order[q]);
        --q;
      }
    }

    // Each player picks a deck.
    decision_class_ = Decisions::kTeamSelect;
    std::vector<IndexT> available_decks{0, 1, 2, 3};
    for (IndexT i = 0; i < 4; ++i) {
      current_player_x_ = pick_order[i];
      IndexT pick;
      // We only get to pick a deck if there's more than one option availble.
      if (i < 3) {
        available_actions_n_ = 4 - i;
        for (IndexT q = 0; q < available_actions_n_; ++q) {
          move_to_policy_i_[q] = available_decks[q];
        }

        // Wait for an answer.
        co_await coroutine::Suspend();
        pick = move_i_;
      } else {
        pick = 0;
      }
      
      // Deal the deck to the player.
      hand_size_[current_player_x_] = 4;
      for (IndexT q = 0; q < 4; ++q) {
        hand_[current_player_x_][q] = kDeckContents[available_decks[pick]][q];
      }

      // Remove the deck from those available.
      available_decks.erase(available_decks.begin() + pick);
    }

    std::array<IndexT, 4> player_selected_index;
    std::array<IndexT, 4> pick_order{0, 1, 2, 3};
    for (current_location_i_ = 0; 
         current_location_i_ < 4;
         ++current_location_i_) {
      // Each player picks a card in a random order.
      std::shuffle(pick_order.begin(), pick_order.end(), bitgen_);

      // The index of the hand card selected by each player 1-4.

      // We only get to pick our cards if there's more than one option availble.
      if (current_location_i_ < 3) {
        decision_class_ = Decisions::kCharacterSelect;
        for (IndexT x : pick_order) {
          current_player_x_ = x;
          available_actions_n_ = hand_size_[x];
          for (IndexT q = 0; q < available_actions_n_; ++q) {
            move_to_policy_i_[q] = hand_[x][q];
          }

          // Wait for an answer.
          co_await coroutine::Suspend();
          player_selected_index[x] = move_i_;
        }
      } else {
        for (IndexT i = 0; i < 4; player_selected_index[i++] = 0);
      }

      // Now remove everyone's selected cards from their hands and put them out.
      // Hand cards need to stay in sorted order as do played cards.

      for (IndexT i = 0; i < 4; ++i) {
        cards_in_play_[i].value = hand_[i][player_selected_index[i]];
        cards_in_play_[i].player_i = i;

        // Remove the card from the player's hand preserving numeric order.
        --hand_size_[i];
        for (IndexT q = player_selected_index[i]; q < hand_size_[i]; ++q) {
          std::swap(hand_[i][q], hand_[i][q + 1]);
        }
      }

      // Sort the played cards.
      std::sort(
          cards_in_play_.begin(),
          cards_in_play_.end(),
          [](PlayedCard a, PlayedCard b) { return a.value < b.value; });

      // Give Bethesda a chance (if she's there, and not alone) to trade places.
      if (current_location_i_ < 3) {
        for (IndexT i = 0; i < 4; ++i) {
          if (cards_in_play_[i].value != kBethesdaIndex) continue;
          decision_class_ = Decisions::kBethesdaSwap;
          current_player_x_ = cards_in_play_[i].player_i;
          available_actions_n_ = hand_size_[current_player_x_] + 1;
          // Move 0 is pick Bethesda (11), other moves are other hand cards.
          move_to_policy_i_[0] = kBethesdaIndex;
          for (IndexT q = 1; q < available_actions_n_; ++q) {
            move_to_policy_i_[q] = hand_[current_player_x_][q - 1];
          }

          // Wait for an answer.
          co_await coroutine::Suspend();

          // No swap.
          if (move_i_ == 0) break;
          
          // Swap!
          cards_in_play_[i].value = hand_[current_player_x_][move_i_ - 1];
          hand_[current_player_x_][move_i_ - 1] = kBethesdaIndex;

          // Re-sort both played cards and hand cards.
          std::sort(
              cards_in_play_.begin(),
              cards_in_play_.end(),
              [](PlayedCard a, PlayedCard b) { return a.value < b.value; });
          std::sort(
              hand_[current_player_x_].begin(),
              hand_[current_player_x_].begin() + hand_size_[current_player_x_]);
        }
      }

      // Now run through the played cards from highest to lowest.

      for (IndexT i = 0; i < 4; ++i) {
        switch (cards_in_play_[i].value) {
          case kMagicianIndex: {



            break;
          }
          default: {
            LOG(FATAL) << "Unknown played card.";
          }
        }
      }






    }




  }
}

}  // namespace ignoble
}  // namespace games
}  // namespace azah
