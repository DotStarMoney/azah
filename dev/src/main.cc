#include <stdint.h>

#include <iostream>
#include <fstream>
#include <string_view>

#include "absl/strings/str_format.h"
#include "games/game.h"
#include "games/ignoble/ignoble4.h"
#include "games/ignoble/ignoble4_network.h"
#include "glog/logging.h"
#include "mcts/rl_player.h"

namespace {

using Game = azah::games::ignoble::Ignoble4;
using GameNetwork = azah::games::ignoble::Ignoble4Network;
using RLPlayer = azah::mcts::RLPlayer<Game, GameNetwork>;

constexpr std::string_view kCheckpointFormat = 
    "c:/usr/azah/checkpoints/ignoble4_%d.dat";

constexpr int kLoadCheckpointIndex = 5250;

constexpr int kCheckpointFreq = 10;

}  // namespace

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);

  RLPlayer player(16);

  RLPlayer::SelfPlayOptions options{
      .learning_rate = 0.01,
      .simulations_n = 1024,
      .root_noise_alpha = 0.6,
      .root_noise_lerp = 0.25,
      .one_hot_breakover_moves_n = 80,
      .exploration_scale = 0.22};

  {
    std::ifstream checkpoint(
        absl::StrFormat(kCheckpointFormat, kLoadCheckpointIndex), 
        std::ios::in | std::ios::binary);
    player.Deserialize(checkpoint);
  }

  for (int i = kLoadCheckpointIndex; i < 10000; ++i) {
    std::cout << "Playing game..." << std::endl;
    auto losses = player.Train(1, options);

    if (((i + 1) % kCheckpointFreq) == 0) {
      std::ofstream checkpoint(absl::StrFormat(kCheckpointFormat, (i + 1)), 
                               std::ios::out | std::ios::binary);
      player.Serialize(checkpoint);
    }

    std::cout << "Finished " << (i + 1) << " games with loss " << losses
        << std::endl;
  }

  return 0;
}
