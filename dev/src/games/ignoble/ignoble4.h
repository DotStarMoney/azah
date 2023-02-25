#ifndef AZAH_GAMES_IGNOBLE_IGNOBLE4_H_
#define AZAH_GAMES_IGNOBLE_IGNOBLE4_H_

#include <stdint.h>

#include <array>
#include <string_view>
#include <vector>

#include "../../nn/data_types.h"
#include "../coroutine.h"
#include "../game.h"
#include "absl/random/random.h"

namespace azah {
namespace games {
namespace ignoble {

class Ignoble4 : public Game<4> {
 public:
  Ignoble4();
  ~Ignoble4();

  const std::string_view name() const override;

  int CurrentPlayerI() const override;
  int CurrentMovesN() const override;
  GameState State() const override;
  std::array<float, 4> Outcome() const override;

  std::vector<nn::DynamicMatrix> StateToMatrix() const override;
  int PolicyClassI() const override;
  float PolicyForMoveI(const nn::DynamicMatrix& policy, 
                       int move_i) const override;

  nn::DynamicMatrix PolicyMask() const override;

  void MakeMove(int move_i) override;

 private:
  static constexpr std::string_view kName_ = "Ignoble 4-Player";
  typedef std::int8_t IndexT;

  // True if player_a should pick before player_b.
  bool ComparePlayerPickOrder(int player_a, int player_b) const;

  absl::BitGen bitgen_;

  // Used to pass messages to RunGame.
  int move_i_;
  coroutine::Void RunGame();
  coroutine::VoidHandle run_handle_;

  bool CheckForWin(IndexT player_x);

  enum class Decisions {
    kUnknown = 0,
    kTeamSelect = 1,
    kCharacterSelect = 2,
    kPrincessStock = 3,
    kMeatBunglerBounty = 4,
    kMerryPiemanStock = 5,
    kBenedictIncrease = 6,
    kBethesdaSwap = 7,
    kOunceStealStock = 8,
    kMagicianStockTakeToss = 9,
    kRepentStock = 10
  };
  Decisions decision_class_;
  
  // A random player order determined at the start of the game that breaks ties
  // for deck selection on [0, 3]. Lower numbers go first.
  std::array<IndexT, 4> deck_select_tie_order_;

  // The hands available to players 1-4. On current_location_i_ = 0, 4 cards are
  // available so indices [0, 3], on current_location_i_ = 1, 3 cards are
  // available so indices [0, 2] and so on. Cards are always in sorted value
  // order in a hand.
  std::array<std::array<IndexT, 4>, 4> hand_;
  // The number of cards held by each player 1-4.
  std::array<IndexT, 4> hand_size_;

  // Broken down by type, the stock for players 1-4. So [0][0] is player 1's
  // soil, [0][1] is player 1's herb etc...
  //
  // Stock types are in order: soil, herb, beast, coin.
  std::array<std::array<IndexT, 4>, 4> stock_n_;

  // The locations dealt, with the current location referenced by
  // current_location_i_
  std::array<IndexT, 4> locations_in_play_;
  IndexT current_location_i_;

  // The shuffled location deck, with top_of_deck_i starting at 11 and working
  // down to 0.
  std::array<IndexT, 12> location_deck_;
  IndexT top_of_deck_i_;

  // The cards played for the current location, sorted from highest to lowest
  // value.
  struct PlayedCard {
    IndexT value;
    IndexT player_i;
  };
  std::array<PlayedCard, 4> cards_in_play_;

  // If The Ounce is stealing, the index of the player who's getting robbed.
  IndexT ounce_hot_seat_;

  // If -1, nobody has won yet. Otherwise, this is the index of the winning
  // player [0, 3].
  IndexT winning_player_i_;

  // Cached values.

  int available_actions_n_;
  int current_player_x_;
  std::array<int, 16> move_to_policy_i_;
};

}  // namespace ignoble
}  // namespace games
}  // namespace azah

#endif  // AZAH_GAMES_IGNOBLE_IGNOBLE4_H_
