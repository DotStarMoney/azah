#ifndef AZAH_GAMES_GAME_H_
#define AZAH_GAMES_GAME_H_

#include <span>
#include <string>
#include <string_view>

namespace azah {
namespace games {

class Game {
 public:
  virtual const std::string_view name() const = 0;

  virtual const int players_n() const = 0;

  virtual const std::string& state_uid() const = 0;

  virtual int CurrentPlayerI() const = 0;
  virtual int CurrentMovesN() const = 0;

  enum class GameState {
    kOngoing = 0,
    kWinner = 1,
    kTie = 2
  };
  virtual GameState State() const = 0;
  
  virtual int WinningPlayerI() const = 0;
  virtual int PolicyToMoveI(const std::span<float>& policy) const = 0;

  virtual void MakeMove(int move_i) = 0;
};

}  // namespace games
}  // namespace azah

#endif  // AZAH_GAMES_GAME_H_
