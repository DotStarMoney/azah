#ifndef AZAH_GAMES_GAME_H_
#define AZAH_GAMES_GAME_H_

#include <array>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "../nn/data_types.h"

namespace azah {
namespace games {

template <int PlayersN, int MaxMoveOptions>
class Game {
 public:
  virtual const std::string_view name() const = 0;

  static constexpr int players_n() { return PlayersN; }

  virtual const std::string& state_uid() const = 0;

  static constexpr int max_move_options_n() { return MaxMoveOptions; }

  virtual int inputs_model_tag() const = 0;
  virtual int target_policies_model_tag() const = 0;
  virtual int target_outcomes_model_tag() const = 0;

  virtual int CurrentPlayerI() const = 0;
  virtual int CurrentMovesN() const = 0;

  enum class GameState {
    kUnknown = 0,
    kOngoing = 1,
    kOver = 2,
  };
  virtual GameState State() const = 0;
  
  virtual std::array<float, PlayersN> Outcome() const = 0;

  virtual std::vector<nn::DynamicMatrix> StateToMatrix() const = 0;
  virtual int PolicyToMoveI(std::span<float const> policy) const = 0;
  virtual int PolicyClassI() const = 0;

  virtual nn::DynamicMatrix MoveVisitCountToPolicy(
      std::span<int const> visits) const = 0;

  virtual void MakeMove(int move_i) = 0;
};

}  // namespace games
}  // namespace azah

#endif  // AZAH_GAMES_GAME_H_
