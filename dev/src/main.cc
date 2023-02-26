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

  int i = 0;
  while (game.State() == azah::games::GameState::kOngoing) {
    int move = absl::Uniform(rng, 0, game.CurrentMovesN());
    game.MakeMove(move, rng);
    ++i;
  }

  std::cout << i;

  return 0;
}
