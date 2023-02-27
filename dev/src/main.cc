#include <iostream>

#include "absl/random/random.h"
#include "games/game.h"
#include "games/ignoble/ignoble4.h"
#include "games/ignoble/ignoble4_network.h"
#include "glog/logging.h"

namespace {

using Game = azah::games::ignoble::Ignoble4;
using Network = azah::games::ignoble::Ignoble4Network;

}  // namespace

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);

  absl::BitGen rng;

  Game game;
  Network net;

  int i = 0;
  while (game.State() == azah::games::GameState::kOngoing) {
    int move = absl::Uniform(rng, 0, game.CurrentMovesN());
    game.MakeMove(move, rng);
    ++i;
  }

  std::cout << i << std::endl;
  auto state = game.StateToMatrix();
  for (int q = 0; q < state.size(); ++q) {
    std::cout << state[q].size() << std::endl;
  }

  return 0;
}
