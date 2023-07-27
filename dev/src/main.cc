#include <stdint.h>

#include <chrono>
#include <iostream>
#include <fstream>
#include <string_view>
#include <vector>

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
constexpr char kStatsFile[] = "c:/usr/azah/checkpoints/chk_stats.txt";

constexpr int kLoadCheckpointIndex = 15000;

constexpr int kCheckpointFreq = 10;

constexpr int kRepeats = 10;

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

  if (kLoadCheckpointIndex > 0)  {
    std::ifstream checkpoint(
        absl::StrFormat(kCheckpointFormat, kLoadCheckpointIndex), 
        std::ios::in | std::ios::binary);
    player.Deserialize(checkpoint);
  }

  Game game;
  absl::BitGen bitgen;

  while (game.State() == azah::games::GameState::kOngoing) {
    std::vector<RLPlayer::EvaluateResult> all_results;
    for (int i = 0; i < kRepeats; ++i) {
      all_results.push_back(player.Evaluate(game, options));
    }

    for (int i = 1; i < kRepeats; ++i) {
      for (int q = 0; q < all_results[0].predicted_move.size(); ++q) {
        all_results[0].predicted_move[q] += all_results[i].predicted_move[q];
      }
      for (int q = 0; q < 4; ++q) {
        all_results[0].predicted_outcome[q] += 
            all_results[i].predicted_outcome[q];
      }
    }
    for (int i = 0; i < all_results[0].predicted_move.size(); ++i) {
      all_results[0].predicted_move[i] /= kRepeats;
    }
    for (int i = 0; i < 4; ++i) {
      all_results[0].predicted_outcome[i] /= kRepeats;
    }

    RLPlayer::EvaluateResult result = std::move(all_results[0]);


    std::cout << "Outcome odds = [";
    for (int i = 0; i < Game::players_n(); ++i) {
      std::cout << result.predicted_outcome[i];
      if (i < (Game::players_n() - 1)) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "Move probability = [";
    int q = -1;
    float q_value = 0.0f;
    for (int i = 0; i < result.predicted_move.size(); ++i) {
      std::cout << result.predicted_move[i];
      if (result.predicted_move[i] > q_value) {
        q_value = result.predicted_move[i];
        q = i;
      }
      if (i < (result.predicted_move.size() - 1)) std::cout << ", ";
    }
    std::cout << "]\n";

    game.MakeMove(q, bitgen);
  }

  /*
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
    {
      std::ofstream stats(kStatsFile, std::ios::out | std::ios::app);
      auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch()).count();
      stats << i << ", " << now << ": " << losses << "\n";
    }
  }
  */
  return 0;
}
