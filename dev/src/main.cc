#include <iostream>

#include "absl/random/random.h"
#include "games/game.h"
#include "games/ignoble/ignoble4.h"
#include "glog/logging.h"

namespace {

using Game = azah::games::ignoble::Ignoble4;

}  // namespace

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);

  absl::BitGen rng;

  Game game;

  //
  //
  //
  // UGH co-routines don't copy cleanly at all (state is compiler dependent),
  // and so we're going to be stuck rolling our own with a jump-table. Sad. Bad.
  //
  //
  //

  while (game.State() == azah::games::GameState::kOngoing) {
    int move = absl::Uniform(rng, 0, game.CurrentMovesN());
    game.MakeMove(move);
  }

  return 0;
}
