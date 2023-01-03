#include <iostream>
#include <vector>

#include "games/game.h"
#include "games/mancala/mancala.h"
#include "games/mancala/mancala_network.h"
#include "glog/logging.h"
#include "mcts/rl_player.h"

namespace {

using Game = azah::games::mancala::Mancala;
using GameNetwork = azah::games::mancala::MancalaNetwork;
using RLPlayer = azah::mcts::RLPlayer<Game, GameNetwork>;

}  // namespace

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);

  RLPlayer player(4);

  RLPlayer::SelfPlayOptions options{
      .learning_rate = 0.01,
      .simulations_n = 128,
      .root_noise_alpha = 0.5,
      .root_noise_lerp = 0.25,
      .one_hot_breakover_moves_n = 5,
      .exploration_scale = 0.25};

  for (int i = 0; i < 100; ++i) {
    std::cout << "Playing games..." << std::endl;
    auto losses = player.Train(1, options);
    std::cout << "Finished " << (i + 1) << " games with loss " << losses 
        << std::endl;
  }

  return 0;
}
