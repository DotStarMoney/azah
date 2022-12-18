#ifndef AZAH_GAMES_GAME_H_
#define AZAH_GAMES_GAME_H_

#include <array>
#include <span>
#include <string>
#include <string_view>

namespace azah {
namespace games {

template <int PlayersN>
class Game {
 public:
  virtual const std::string_view name() const = 0;

  virtual const int players_n() const = 0;

  virtual const std::string& state_uid() const = 0;

  virtual int CurrentPlayerI() const = 0;
  virtual int CurrentMovesN() const = 0;

  enum class GameState {
    kUnknown = 0,
    kOngoing = 1,
    kOver = 2,
  };
  virtual GameState State() const = 0;
  
  virtual std::array<float, PlayersN> Outcome() const = 0;

  virtual void StateToVector(std::span<float> out) const = 0;
  virtual int PolicyToMoveI(std::span<float const> policy) const = 0;
  virtual int PolicyClassI() const = 0;

  virtual void MakeMove(int move_i) = 0;
};

}  // namespace games
}  // namespace azah

#endif  // AZAH_GAMES_GAME_H_
