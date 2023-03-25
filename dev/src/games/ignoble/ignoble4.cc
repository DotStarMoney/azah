#include "ignoble4.h"

#include <stddef.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <string_view>
#include <vector>

#include "../../nn/data_types.h"
#include "../../nn/init.h"
#include "../game.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"
#include "glog/logging.h"

// I'm so sick of people doing this, it's 2023. Stop.
#undef max
#undef min

namespace azah {
namespace games {
namespace ignoble {
namespace {

// The stock state offsets in order of soil, herb, beast, coin.
constexpr int kStockStateOffsets[] = {0, 23, 46, 69};

constexpr int kHandCardsOffset = 92;
constexpr int kPlayerCardOffset = 108;
constexpr int kTieOrderOffset = 124;
constexpr int kOunceHotSeatOffset = 128;

constexpr int kLocationOffset = 0;
constexpr int kRemainingLocationsOffset = 48;

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
        true,   // Meat Bungler toss
        true,   // Meat Bungler stock
        true,   // Merry Pieman
        true,   // Benedict
        true,   // Bethesda
        true,   // Ounce
        true,   // Magician
        false   // Repent
    };

// The size of the column vector representing an individual decision.
constexpr int kMaxDecisionRowsN[] = {
        0,   // Unknown
        4,   // Selecting team 
        16,  // Selecting character
        4,   // Princess
        2,   // Meat Bungler toss
        2,   // Meat Bungler stock
        4,   // Merry Pieman
        2,   // Benedict
        2,   // Bethesda 
        4,   // Ounce
        8,   // Magician
        5    // Repent
    };

struct Location {
  // 0=soil, 1=herb, 2=beast, 3=coin
  int type;
  int bounty_n;
};
constexpr Location kLocations[] = {
        {0, 2},  // 0 Soot Cellar
        {0, 3},  // 1 Swamp
        {0, 3},  // 2 Peasant Village
        {1, 1},  // 3 The Hedgerow
        {1, 2},  // 4 Grove
        {1, 3},  // 5 Vegetable Garden
        {2, 1},  // 6 Thy Frigid Plunge
        {2, 2},  // 7 Pig Sty
        {2, 2},  // 8 Plains
        {3, 1},  // 9 Market
        {3, 1},  // 10 The Tomb
        {3, 2}   // 11 The Coiniary
    };

constexpr std::array<std::array<int, 4>, 4> kDeckContents = {{
        {2, 7, 9, 12},   // The Ounce
        {0, 6, 11, 13},  // The King
        {3, 5, 8, 14},   // The Magician
        {1, 4, 10, 15}   // Death
    }};

// Labels for the horrid jump table.
constexpr int kJumpInit = -1;
constexpr int kJumpTeamSelect = 0;
constexpr int kJumpCharacterSelect = 1;
constexpr int kJumpPrincessStock = 2;
constexpr int kJumpMeatBunglerToss = 3;
constexpr int kJumpMeatBunglerStock = 4;
constexpr int kJumpMerryPiemanStock = 5;
constexpr int kJumpBenedictIncrease = 6;
constexpr int kJumpBethesdaSwap = 7;
constexpr int kJumpOunceStealStock = 8;
constexpr int kJumpMagicianStockTakeToss = 9;
constexpr int kJumpRepentStock = 10;

// After 4050 decisions, it's a tie. If you're wondering where this comes from,
// it's the ~80th percentile of average game depths over 10000 purely random
// games.
constexpr int kMaxDepth = 4050;

template <std::signed_integral IntType>
IntType bounty_value(IntType stock_value, IntType modifier) {
  return std::max(stock_value + modifier, 0);
}

}  // namespace

Ignoble4::Ignoble4() :
    jump_label_(kJumpInit),
    decision_class_(Decisions::kTeamSelect),
    deck_select_tie_order_{0, 1, 2, 3},
    hand_size_{0, 0, 0, 0},
    stock_n_{{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
    current_location_i_(0),
    location_deck_{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
    top_of_deck_i_(-1),
    winning_player_i_(-1),
    out_of_time_(false),
    out_of_time_outcome_{-1, -1, -1, -1},
    depth_(0) {
  // Just for the first shuffle.
  absl::BitGen bitgen;
  MakeMove(-1, bitgen);
}

Ignoble4::Ignoble4(const std::vector<int>& fixed_deck_select_tie_order) :
    jump_label_(kJumpInit),
    fixed_deck_select_tie_order_(fixed_deck_select_tie_order),
    decision_class_(Decisions::kTeamSelect),
    deck_select_tie_order_{0, 1, 2, 3},
    hand_size_{0, 0, 0, 0},
    stock_n_{{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
    current_location_i_(0),
    location_deck_{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
    top_of_deck_i_(-1),
    winning_player_i_(-1),
    out_of_time_(false),
    out_of_time_outcome_{-1, -1, -1, -1},
    depth_(0) {
  // Just for the first shuffle.
  absl::BitGen bitgen;
  MakeMove(-1, bitgen);
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
  if (out_of_time_) {
    if (out_of_time_outcome_[0] == -1) {
      float sum = 0.0f;
      for (int i = 0; i < 4; ++i) {
        // The amount we still need...
        int total_missing = 0;
        int total = 0;
        for (int q = 0; q < 4; ++q) {
          total_missing += std::max(5 - stock_n_[i][q], 0);
          total += stock_n_[i][q];
        }
        // The amount we'd have to throw away to get there...
        int total_extra = std::max(total_missing - (22 - total), 0);
        out_of_time_outcome_[i] = 
            1.0f / static_cast<float>(total_missing + total_extra);
        sum += out_of_time_outcome_[i];
      }
      for (int i = 0; i < 4; ++i) {
        out_of_time_outcome_[i] = out_of_time_outcome_[i] / sum;
      }
    }
    return out_of_time_outcome_;
  } else {
    std::array<float, 4> outcome{0.0f, 0.0f, 0.0f, 0.0f};
    outcome[winning_player_i_] = 1.0f;
    return outcome;
  }
}

std::vector<nn::DynamicMatrix> Ignoble4::StateToMatrix() const {
  // This provides 5 inputs:
  //   - 4x player state as a 129-D vector.
  //   - 1x global state as a 60-D vector.
  // 
  // The player currently making a decision is rotated into the first slot.
  std::vector<nn::DynamicMatrix> inputs;
  for (std::size_t i = 0; i < 4; ++i) {
    inputs.push_back(nn::init::Zeros<129, 1>());
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
      f(kOunceHotSeatOffset) = 1.0f;
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
  for (int card_i = 0; card_i <= top_of_deck_i_; ++card_i) {
    g(kRemainingLocationsOffset + location_deck_[card_i], 0) = 1.0f;
  }
 
  return inputs;
}

int Ignoble4::PolicyClassI() const {
  DCHECK_NE(static_cast<int>(decision_class_), 
            static_cast<int>(Decisions::kUnknown));
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
  DCHECK_NE(static_cast<int>(decision_class_),
            static_cast<int>(Decisions::kUnknown));
  nn::DynamicMatrix mask(
      kMaxDecisionRowsN[static_cast<int>(decision_class_)], 1);
  mask.setZero();
  for (std::size_t i = 0; i < available_actions_n_; ++i) {
    mask(move_to_policy_i_[i], 0) = 1.0f;
  }
  return mask;
}

bool Ignoble4::CheckForWin(IndexT player_x) {
  for (std::size_t i = 0; i < 4; ++i) {
    if (stock_n_[player_x][i] < 5) return false;
  }
  winning_player_i_ = player_x;
  return true;
}

bool Ignoble4::PlayerFull(IndexT player_x) const {
  IndexT total_stock = 0;
  for (IndexT q = 0; q < 4; ++q) {
    total_stock += stock_n_[player_x][q];
    if (total_stock >= 22) return true;
  }
  return false;
}

void Ignoble4::MakeMove(int move_i, absl::BitGenRef bitgen) {
  depth_ += 1;
  if (depth_ > kMaxDepth) {
    winning_player_i_ = 0;
    out_of_time_ = true;
    return;
  }

  // So a word on how we got here: first I implemented this as a coroutine, and
  // it was clean and lovely. Then I found out C++ really doesn't want to give
  // you access to the underlying state in a cross-platform way, but by then I
  // had already realized how much cleaner it is to implement games using
  // asynchronous transfer... so I split the difference and made a fauxroutine.
  //
  // That being said, I'm so mad at C++ and MSVC. Coroutines aren't copyable, 
  // dynamic jump tables aren't possible since label addresses aren't storable, 
  // all fine features that just don't exist... so here we are... jumping
  // twice... (yes this will get jump threaded during optimization, still dumb).
  switch (jump_label_) {
  case kJumpTeamSelect: goto MakeMove_TeamSelect;
  case kJumpCharacterSelect: goto MakeMove_CharacterSelect;
  case kJumpPrincessStock: goto MakeMove_PrincessStock;
  case kJumpMeatBunglerToss: goto MakeMove_MeatBunglerToss;
  case kJumpMeatBunglerStock: goto MakeMove_MeatBunglerStock;
  case kJumpMerryPiemanStock: goto MakeMove_MerryPiemanStock;
  case kJumpBenedictIncrease: goto MakeMove_BenedictIncrease;
  case kJumpBethesdaSwap: goto MakeMove_BethesdaSwap;
  case kJumpOunceStealStock: goto MakeMove_OunceStealStock;
  case kJumpMagicianStockTakeToss: goto MakeMove_MagicianStockTakeToss;
  case kJumpRepentStock: goto MakeMove_RepentStock;
  default:;
  }

  // This will never change, and is the equivalent of picking a player to start
  // and the seating order at the table.
  if (fixed_deck_select_tie_order_.empty()) {
    std::shuffle(deck_select_tie_order_.begin(), deck_select_tie_order_.end(),
                 bitgen);
  } else {
    for (int i = 0; i < 4; ++i) {
      deck_select_tie_order_[i] = fixed_deck_select_tie_order_[i];
    }
  }
  for (;;) {
    // Start of a new round. First we check to see if a location shuffle is in
    // order.
    if (top_of_deck_i_ == -1) {
      std::shuffle(location_deck_.begin(), location_deck_.end(), bitgen);
      top_of_deck_i_ = 11;
    }

    // Deal the four locations.
    for (s_.i = 0; s_.i < 4; ++s_.i) {
      locations_in_play_[s_.i] = location_deck_[top_of_deck_i_ - s_.i];
    }
    top_of_deck_i_ -= 4;

    // Figure out the pick order.
    s_.pick_order = {0, 1, 2, 3};
    for (s_.i = 1; s_.i < 4; ++s_.i) {
      s_.q = s_.i - 1;
      while ((s_.q >= 0) && ComparePlayerPickOrder(s_.pick_order[s_.q + 1], 
                                                   s_.pick_order[s_.q])) {
        std::swap(s_.pick_order[s_.q + 1], s_.pick_order[s_.q]);
        --s_.q;
      }
    }

    // Each player picks a deck.
    decision_class_ = Decisions::kTeamSelect;
    s_.available_decks = {0, 1, 2, 3};
    for (s_.i = 0; s_.i < 4; ++s_.i) {
      current_player_x_ = s_.pick_order[s_.i];

      // We only get to pick a deck if there's more than one option availble.
      if (s_.i < 3) {
        available_actions_n_ = 4 - s_.i;
        for (s_.q = 0; s_.q < available_actions_n_; ++s_.q) {
          move_to_policy_i_[s_.q] = s_.available_decks[s_.q];
        }

        // Wait for an answer.
        jump_label_ = kJumpTeamSelect; return;
MakeMove_TeamSelect:
        s_.pick = move_i;
      } else {
        s_.pick = 0;
      }
      
      // Deal the deck to the player.
      hand_size_[current_player_x_] = 4;
      for (s_.q = 0; s_.q < 4; ++s_.q) {
        hand_[current_player_x_][s_.q] = 
            kDeckContents[s_.available_decks[s_.pick]][s_.q];
      }

      // Remove the deck from those available.
      for (s_.q = s_.pick; s_.q < (3 - s_.i); ++s_.q) {
        std::swap(s_.available_decks[s_.q], s_.available_decks[s_.q + 1]);
      }
    }

    // We use this to track if someone took stock, since if they did, they don't
    // get to repent.
    s_.repent_check = {true, true, true, true};
    for (current_location_i_ = 0; 
         current_location_i_ < 4;
         ++current_location_i_) {
      // Each player select a card in a random order.
      if (fixed_select_order_.empty()) {
        s_.select_order = {0, 1, 2, 3};
        std::shuffle(s_.select_order.begin(), s_.select_order.end(), bitgen);
      } else {
        for (int i = 0; i < 4; ++i) {
          s_.select_order[i] = fixed_select_order_[i];
        }
      }

      // The index of the hand card selected by each player 1-4.

      // We only get to pick our cards if there's more than one option availble.
      if (current_location_i_ < 3) {
        decision_class_ = Decisions::kCharacterSelect;
        for (s_.i = 0; s_.i < 4; ++s_.i) {
          s_.x = s_.select_order[s_.i];
          current_player_x_ = s_.x;
          available_actions_n_ = hand_size_[s_.x];
          for (s_.q = 0; s_.q < available_actions_n_; ++s_.q) {
            move_to_policy_i_[s_.q] = hand_[s_.x][s_.q];
          }

          // Wait for an answer.
          jump_label_ = kJumpCharacterSelect; return;
MakeMove_CharacterSelect:
          s_.player_selected_index[s_.x] = move_i;
        }
      } else {
        for (s_.i = 0; s_.i < 4; s_.player_selected_index[s_.i++] = 0);
      }

      // Now remove everyone's selected cards from their hands and put them out.
      // Hand cards need to stay in sorted order as do played cards.

      for (s_.i = 0; s_.i < 4; ++s_.i) {
        cards_in_play_[s_.i].value = 
            hand_[s_.i][s_.player_selected_index[s_.i]];
        cards_in_play_[s_.i].player_i = s_.i;

        // Remove the card from the player's hand preserving numeric order.
        --hand_size_[s_.i];
        for (s_.q = s_.player_selected_index[s_.i]; 
             s_.q < hand_size_[s_.i]; 
             ++s_.q) {
          std::swap(hand_[s_.i][s_.q], hand_[s_.i][s_.q + 1]);
        }
      }

      // Sort the played cards.
      std::sort(
          cards_in_play_.begin(),
          cards_in_play_.end(),
          [](PlayedCard a, PlayedCard b) { return a.value < b.value; });

      // Give Bethesda a chance (if she's still in someone's hand) to trade
      // places.
      if (current_location_i_ < 3) {
        for (s_.i = 0; s_.i < 4; ++s_.i) {
          s_.bethesda_hand_i = -1;
          for (s_.q = 0; s_.q < hand_size_[s_.i]; ++s_.q) {
            if (hand_[s_.i][s_.q] == kBethesdaIndex) {
              s_.bethesda_hand_i = s_.q;
              break;
            }
          }
          if (s_.bethesda_hand_i == -1) continue;
          current_player_x_ = s_.i;
          for (s_.q = 0; s_.q < 4; ++s_.q) {
            if (cards_in_play_[s_.q].player_i == current_player_x_) {
              s_.played_card_i = s_.q;
              break;
            }
          }

          // bethesda_hand_i is the index in current_player_x_'s hand that has
          // Bethesda, and played_card_i is in the index in cards_in_play_ that
          // has the card that would be swapped with Bethesda.

          decision_class_ = Decisions::kBethesdaSwap;
          available_actions_n_ = 2;
          // Move 0 is stay, otherwise swap.
          move_to_policy_i_[0] = 0;
          move_to_policy_i_[1] = 1;

          // Wait for an answer.
          jump_label_ = kJumpBethesdaSwap; return;
MakeMove_BethesdaSwap:
          // No swap.
          if (move_i == 0) break;
          
          // Swap!
          std::swap(cards_in_play_[s_.played_card_i].value,
                    hand_[current_player_x_][s_.bethesda_hand_i]);

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
      s_.loc_i = locations_in_play_[current_location_i_];
      s_.stock_modifier = 0;
      s_.bungler_tossed = false;
      for (s_.i = 3; s_.i >= 0; --s_.i) {
        current_player_x_ = cards_in_play_[s_.i].player_i;
        switch (cards_in_play_[s_.i].value) {
          case kMagicianIndex: {
            // If we didn't win, we don't get to use the ability.
            if (s_.i < 3) break;

            decision_class_ = Decisions::kMagicianStockTakeToss;

            // First determine what is possible. We can only toss stock we have,
            // and only take when we're not full.
        
            s_.full = PlayerFull(current_player_x_);
            available_actions_n_ = s_.full ? 0 : 4;

            s_.tossable_types_n = 0;
            for (s_.q = 0; s_.q < 4; ++s_.q) {
              if (stock_n_[current_player_x_][s_.q] == 0) continue;
              s_.tossable_types[s_.tossable_types_n++] = s_.q;
              ++available_actions_n_;
            }
            
            // First 4 actions are for tossing, last 4 actions are for taking.
            for (s_.q = 0; s_.q < s_.tossable_types_n; ++s_.q) {
              move_to_policy_i_[s_.q] = s_.tossable_types[s_.q];
            }

            if (!s_.full) {
              for (s_.q = 0; s_.q < 4; ++s_.q) {
                move_to_policy_i_[s_.tossable_types_n + s_.q] = 4 + s_.q;
              }
            }
            
            // Wait for an answer.
            jump_label_ = kJumpMagicianStockTakeToss; return;
MakeMove_MagicianStockTakeToss:

            if (move_i >= s_.tossable_types_n) {
              // We took one.
              s_.type = move_i - s_.tossable_types_n;
              ++stock_n_[current_player_x_][s_.type];
              s_.repent_check[current_player_x_] = false;
              if (CheckForWin(current_player_x_)) return;
            } else {
              // We tossed one.
              s_.type = s_.tossable_types[move_i];
              --stock_n_[current_player_x_][s_.type];
            }

            break;
          }
          case kOunceIndex: {
            decision_class_ = Decisions::kOunceStealStock;
            // For each player we can steal from, from highest card to lowest...
            for (s_.q = 3; s_.q > s_.i; --s_.q) {
              // Leave if we're at capacity.
              if (PlayerFull(current_player_x_)) break;
              ounce_hot_seat_ = cards_in_play_[s_.q].player_i;

              // Lets see what's available...
              available_actions_n_ = 0;
              for (s_.s = 0; s_.s < 4; ++s_.s) {
                if (stock_n_[ounce_hot_seat_][s_.s] == 0) continue;
                move_to_policy_i_[available_actions_n_++] = s_.s;
              }
              // Player is poor, move on to the next sucker.
              if (available_actions_n_ == 0) continue;

              if (available_actions_n_ > 1) {
                // Wait for an answer.
                jump_label_ = kJumpOunceStealStock; return;
MakeMove_OunceStealStock:
                s_.type = move_to_policy_i_[move_i];
              } else {
                // If they only have one type to filch, just take that.
                s_.type = move_to_policy_i_[0];
              }

              // Perform the steal.
              --stock_n_[ounce_hot_seat_][s_.type];
              ++stock_n_[current_player_x_][s_.type];
              s_.repent_check[current_player_x_] = false;
              if (CheckForWin(current_player_x_)) return;
            }
            break;
          }
          case kSirStrawberryIndex: {
            // This one is easy, if we can take one of the current stock, do it.
            if (PlayerFull(current_player_x_)) break;
            ++stock_n_[current_player_x_][kLocations[s_.loc_i].type];
            s_.repent_check[current_player_x_] = false;
            if (CheckForWin(current_player_x_)) return;
            break;
          }
          case kBenedictIndex: {
            // Raise the bounty by 2 or don't.
            decision_class_ = Decisions::kBenedictIncrease;
            available_actions_n_ = 2;
            move_to_policy_i_[0] = 0;
            move_to_policy_i_[1] = 1;

            // Wait for an answer.
            jump_label_ = kJumpBenedictIncrease; return;
MakeMove_BenedictIncrease:

            if (move_i == 1) {
              s_.stock_modifier += 2;
            }

            break;
          }
          case kPiemanIndex: {
            // If we did win (weird), we don't get to use the ability.
            // If we're full, can't take anything.
            if ((s_.i == 3) || PlayerFull(current_player_x_)) break;

            decision_class_ = Decisions::kMerryPiemanStock;
            available_actions_n_ = 4;
            for (s_.q = 0; s_.q < 4; ++s_.q) move_to_policy_i_[s_.q] = s_.q;

            // Wait for an answer.
            jump_label_ = kJumpMerryPiemanStock; return;
MakeMove_MerryPiemanStock:

            ++stock_n_[current_player_x_][move_i];
            s_.repent_check[current_player_x_] = false;
            if (CheckForWin(current_player_x_)) return;
            break;
          }
          case kBunglerIndex: {
            s_.bounty_value = bounty_value(
                static_cast<IndexT>(kLocations[s_.loc_i].bounty_n), 
                s_.stock_modifier);

            // If there's no bounty, we can't do anything.
            if (s_.bounty_value <= 0) break;

            // If we have nothing in the current stock type, we also can't do
            // anything.
            if (stock_n_[current_player_x_][kLocations[s_.loc_i].type] == 0) {
              break;
            }

            decision_class_ = Decisions::kMeatBunglerToss;
            // We define the action vector as [do nothing, toss it].
            available_actions_n_ = 2;
            move_to_policy_i_[0] = 0;
            move_to_policy_i_[1] = 1;

            // Wait for an answer.
            jump_label_ = kJumpMeatBunglerToss; return;
MakeMove_MeatBunglerToss:

            if (move_i == 1) {
              auto& stock = 
                  stock_n_[current_player_x_][kLocations[s_.loc_i].type];
              stock -= std::max(stock - s_.bounty_value, 0);
              s_.bungler_tossed = true;
            }

            break;
          }
          case kPinderIndex: {
            // If we're full, can't take anything.
            if (PlayerFull(current_player_x_)) break;
            // If The Ounce was played, take a free one.
            for (s_.q = s_.i + 1; s_.q < 4; ++s_.q) {
              if (cards_in_play_[s_.q].value == kOunceIndex) {
                ++stock_n_[current_player_x_][kLocations[s_.loc_i].type];
                s_.repent_check[current_player_x_] = false;
                if (CheckForWin(current_player_x_)) return;
                break;
              }
            }
            break;
          }
          case kBroomemanIndex: {
            // If we did win, we don't get to use the ability.
            // If we're full, can't take anything.
            if ((s_.i == 3) || PlayerFull(current_player_x_)) break;
            // If there's at least 3 stock here after modifiers, take a free
            // one.
            if ((kLocations[s_.loc_i].bounty_n + s_.stock_modifier) >= 3) {
              ++stock_n_[current_player_x_][kLocations[s_.loc_i].type];
              s_.repent_check[current_player_x_] = false;
              if (CheckForWin(current_player_x_)) return;
            }
            break;
          }
          case kKnaveIndex: {
            // Dec the stock modifier, that's it!
            --s_.stock_modifier;
            break;
          }
          case kTavernFoolIndex: {
            // Toss one of the current stock if we can.
            if (stock_n_[current_player_x_][kLocations[s_.loc_i].type] == 0) {
              break;
            }
            --stock_n_[current_player_x_][kLocations[s_.loc_i].type];
            break;
          }
          default: {
            // The card has no effect when resolving order.
          }
        }
      }

      // Resolve taking the bounty.

      s_.bounty_value = bounty_value(
          static_cast<IndexT>(kLocations[s_.loc_i].bounty_n),
          s_.stock_modifier);
      s_.bounty_type = kLocations[s_.loc_i].type;

      current_player_x_ = cards_in_play_[3].player_i;
      switch (cards_in_play_[3].value) {
        case kKingIndex: {
          // The King always takes as if it were worth 2.
          s_.bounty_value = 2;
          break;
        }
        case kPiemanIndex: {
          // The pieman doesn't get to take at all (weird).
          s_.bounty_value = 0;
          break;
        }
        case kBunglerIndex: {
          // If we already tossed, there's no bounty, or we're full, there's
          // nothing to be done.
          if (s_.bungler_tossed
              || (s_.bounty_value == 0)
              || (PlayerFull(current_player_x_))) {
            break;
          }

          decision_class_ = Decisions::kMeatBunglerStock;

          // We define the action vector as [take it (do nothing), ignore it].
          available_actions_n_ = 2;
          move_to_policy_i_[0] = 0;
          move_to_policy_i_[1] = 1;

          // Wait for an answer.
          jump_label_ = kJumpMeatBunglerStock; return;
MakeMove_MeatBunglerStock:

          if (move_i == 1) s_.bounty_value = 0;

          break;
        }
        case kPrincessIndex: {
          // If there's no bounty, or we're full, we can't do anything.
          if ((s_.bounty_value <= 0) || PlayerFull(current_player_x_)) break;
          decision_class_ = Decisions::kPrincessStock;
          available_actions_n_ = 4;
          for (s_.i = 0; s_.i < 4; ++s_.i) move_to_policy_i_[s_.i] = s_.i;

          // Wait for an answer.
          jump_label_ = kJumpPrincessStock; return;
MakeMove_PrincessStock:

          s_.bounty_type = move_i;
          break;
        }
        case kPageboyIndex: {
          // The Pageboy always takes as if it were worth 6, lol.
          s_.bounty_value = 6;
          break;
        }
        default: {
          // The card has no effect on the bounty. 
        }
      }

      s_.total_stock = 0;
      for (s_.i = 0; s_.i < 4; ++s_.i) {
        s_.total_stock += stock_n_[current_player_x_][s_.i];
      }
      // Cap the bounty on the storage limit.
      s_.adj_bounty_value = std::min(static_cast<IndexT>(22 - s_.total_stock),
                                     s_.bounty_value);
      if (s_.adj_bounty_value > 0) {
        // Stock was taken.
        stock_n_[current_player_x_][s_.bounty_type] += s_.adj_bounty_value;
        s_.repent_check[current_player_x_] = false;
        if (CheckForWin(current_player_x_)) return;
      }
 
      // That's a hand, on to the next one!
    }

    // Repent in a random order.
    if (fixed_repent_order_.empty()) {
      s_.repent_order = {0, 1, 2, 3};
      std::shuffle(s_.repent_order.begin(), s_.repent_order.end(), bitgen);
    } else {
      for (int i = 0; i < 4; ++i) {
        s_.repent_order[i] = fixed_repent_order_[i];
      }
    }

    // Before we go to the next round, determine which players may repent for
    // their sins.
    decision_class_ = Decisions::kRepentStock;
    for (s_.i = 0; s_.i < 4; ++s_.i) {
      current_player_x_ = s_.repent_order[s_.i];
      if (!s_.repent_check[current_player_x_]) continue;

      // Determine what could be tossed.
      available_actions_n_ = 1;
      move_to_policy_i_[0] = 0;
      for (s_.q = 0; s_.q < 4; ++s_.q) {
        if (stock_n_[current_player_x_][s_.q] == 0) continue;
        move_to_policy_i_[available_actions_n_++] = 1 + s_.q;
      }

      // We're poor, we took nothing, and so we are not worthy of atonement.
      if (available_actions_n_ == 1) continue;

      // Wait for an answer.
      jump_label_ = kJumpRepentStock; return;
MakeMove_RepentStock:

      if (move_i == 0) continue;
      s_.type = move_to_policy_i_[move_i] - 1;
      --stock_n_[current_player_x_][s_.type];
    }

    // That's a round, on to the next one!
  }
}

void Ignoble4::SetLocations(const std::vector<int>& in_play,
                            const std::vector<int>& deck) {
  for (auto in_play_card : in_play) {
    for (auto deck_card : deck) {
      CHECK_NE(in_play_card, deck_card);
    }
  }
  CHECK_EQ(in_play.size(), 4 - current_location_i_);
  CHECK_EQ(deck.size() - 1, top_of_deck_i_);

  for (int i = 0, q = current_location_i_; i < in_play.size(); ++i, ++q) {
    locations_in_play_[i] = in_play[q];
  }
  for (int i = 0; i < top_of_deck_i_ + 1; ++i) {
    location_deck_[i] = deck[i];
  }
}

void Ignoble4::SetFixedSelectOrder(const std::vector<int>& order) {
  CHECK_EQ(order.size(), 4);
  fixed_select_order_.resize(4);
  for (int i = 0; i < 4; ++i) {
    fixed_select_order_[i] = order[i];
  }
}

void Ignoble4::SetFixedRepentOrder(const std::vector<int>& order) {
  CHECK_EQ(order.size(), 4);
  fixed_repent_order_.resize(4);
  for (int i = 0; i < 4; ++i) {
    fixed_repent_order_[i] = order[i];
  }
}

}  // namespace ignoble
}  // namespace games
}  // namespace azah
