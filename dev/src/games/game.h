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

  virtual int current_player_i() const = 0;
  virtual int current_moves_n() const = 0;
  virtual bool game_over() const = 0;
  virtual int winning_player_i() const = 0;
  virtual int PolicyToMoveI(const std::span<float>& policy) const = 0;

  virtual Game MakeMove(int move_i) = 0;
};

}  // namespace games
}  // namespace azah

#endif  // AZAH_GAMES_GAME_H_
