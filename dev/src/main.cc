#include <iostream>

#include "games/game.h"
#include "games/ignoble/ignoble4.h"
#include "games/ignoble/ignoble4_network.h"
#include "glog/logging.h"
#include "mcts/rl_player.h"

namespace {

using Game = azah::games::ignoble::Ignoble4;
using GameNetwork = azah::games::ignoble::Ignoble4Network;
using RLPlayer = azah::mcts::RLPlayer<Game, GameNetwork>;

}  // namespace

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);

  RLPlayer player(1);

  RLPlayer::SelfPlayOptions options{
      .learning_rate = 0.01,
      .simulations_n = 1024,
      .root_noise_alpha = 0.6,
      .root_noise_lerp = 0.25,
      .one_hot_breakover_moves_n = 80,
      .exploration_scale = 0.25 };

  for (int i = 0; i < 100; ++i) {
    std::cout << "Playing games..." << std::endl;
    auto losses = player.Train(1, options);
    std::cout << "Finished " << (i + 1) << " games with loss " << losses
        << std::endl;
  }

  return 0;
}
