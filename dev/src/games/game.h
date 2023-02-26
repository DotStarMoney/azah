#ifndef AZAH_GAMES_GAME_H_
#define AZAH_GAMES_GAME_H_

#include <array>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include "../nn/data_types.h"
#include "absl/random/random.h"

namespace azah {
namespace games {

enum class GameState {
  kUnknown = 0,
  kOngoing = 1,
  kOver = 2,
};

template <int PlayersN>
class Game {
 public:
  // The canonical name for this game.
  virtual const std::string_view name() const = 0;

  // The number of players in this game. The size of the array returned by
  // Outcome is this long.
  static constexpr int players_n() { return PlayersN; }

  // The index of the player who's move it is to make.
  //
  // Undefined if the game is over.
  virtual int CurrentPlayerI() const = 0;

  // The number of move options the current player has.
  //
  // Undefined if the game is over.
  virtual int CurrentMovesN() const = 0;

  // The state of the current game.
  virtual GameState State() const = 0;
  
  // An array as long as players_n() with the outcome. This array should sum to
  // 1. The order of players in the outcome should match what is returned by
  // CurrentPlayerI(). It is okay to return fractional values in the case of a 
  // tie.
  // 
  // The player who's outcome value is the highest is considered the winner.
  //
  // Undefined if the game is ongoing.
  virtual std::array<float, PlayersN> Outcome() const = 0;

  // Converts the current game state to a series of matrices. These must be 1:1
  // with the associated game model's inputs and will be used to compute the
  // outcome and policy at a given state.
  //
  // This operation will be called once per MCTS node expansion, and when 
  // building the move list. Since the former is the expensive part, there isn't
  // much benefit to caching the value this returns.
  //
  // <!> The created state should reflect the player who's decision it is to 
  //     make such that outcomes can be predicted in rotated "current player"
  //     order. So in a 3 player game, if its player 2's turn, the model should
  //     be able to report outcomes (from these inputs) for player 2, 3, 1 in
  //     that order. For player 3's turn, 3, 1, 2 and for player 1's turn 1, 2,
  //     3. This need not reflect decision order, just *who* is currently making
  //     a decision in a fixed ordering.
  // 
  //     E.g.: In the case of Tic-tac-toe, the player who's move it is could
  //     have their positions represented as 1s and the opponent as -1s instead
  //     of 1s for Xs and -1s for Os.
  //
  //     E.g.: For Chinese checkers when using six matrices to describe the 6
  //     player's board positions, the order of these matrices as returned by
  //     StateToMatrix() should be rotated such that CurrentPlayerI()'s board
  //     matrix is always in the first array position.
  //
  virtual std::vector<nn::DynamicMatrix> StateToMatrix() const = 0;

  // The index of the policy head in the associated game model for the current
  // decision to be made in this game.
  //
  // Undefined if the game is over.
  virtual int PolicyClassI() const = 0;

  // For the given move option index and output from the associated game model's
  // policy head (dictated by PolicyClassI()), extract the value from the policy
  // head associated with the move option index. Policy heads return column
  // vectors.
  //
  // Undefined if the game is over.
  virtual float PolicyForMoveI(const nn::DynamicMatrix& policy, 
                               int move_i) const = 0;

  // Return a 0/1 mask for valid moves compatible with the policy head on the
  // associated game model dictated by PolicyClassI().
  //
  // Undefined if the game is over.
  virtual nn::DynamicMatrix PolicyMask() const = 0;

  // Game subclasses must implement one of the following:
  //
  // 1.)
  // 
  // Make the move for the given index deterministically.
  //
  // Undefined if the game is over.
  // virtual void MakeMove(int move_i) = 0;
  //
  // 2.)
  // 
  // Make the move for the given index potentially with random events.
  //
  // Undefined if the game is over.
  // virtual void MakeMove(int move_i, absl::BitGenRef bitgen) = 0;
};

namespace internal {

template <typename T>
concept GameType =
    requires { T::players_n(); }
    && std::is_base_of<Game<T::players_n()>, T>::value;

}  // namespace internal

template <typename T>
concept DeterministicGameType = 
    internal::GameType<T> && requires(T& game) { game.MakeMove(0); };

template <typename T>
concept NonDeterministicGameType =
    internal::GameType<T> && requires(T & game) {
        game.MakeMove(0, absl::BitGen()); };

template <typename T>
concept AnyGameType = DeterministicGameType<T> || NonDeterministicGameType<T>;

}  // namespace games
}  // namespace azah

#endif  // AZAH_GAMES_GAME_H_
