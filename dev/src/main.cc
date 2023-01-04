#include <stdlib.h>

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
      .simulations_n = 256,
      .root_noise_alpha = 0.5,
      .root_noise_lerp = 0.25,
      .one_hot_breakover_moves_n = 5,
      .exploration_scale = 0.25};

  for (int i = 0; i < 2000; ++i) {
    std::cout << "Playing games..." << std::endl;
    auto losses = player.Train(1, options);
    std::cout << "Finished " << (i + 1) << " games with loss " << losses 
        << std::endl;
  }
  
  Game game;
  int turn_i = 1;
  while (game.State() == azah::games::GameState::kOngoing) {
    std::cout << "----------------------------------------------------------\n";
    std::cout << "Turn " << turn_i << ": PLAYER " 
        << ((game.CurrentPlayerI() == 0) ? "A" : "B") << "\n";
    std::cout << "----------------------------------------------------------\n";

    auto eval_results = player.Evaluate(game, options);

    std::cout << "Predicted outcome: [" << eval_results.predicted_outcome[0]
        << ", " << eval_results.predicted_outcome[1] << "]\n";
    
    int max_i = 0;
    float max_v = -1;
    std::cout << "Predicted moves: [";
    for (int i = 0; i < game.CurrentMovesN(); ++i) {
      std::cout << eval_results.predicted_move[i] << 
          ((i < (game.CurrentMovesN() - 1)) ? ", " : "]\n");
      if (eval_results.predicted_move[i] > max_v) {
        max_v = eval_results.predicted_move[i];
        max_i = i;
      }
    }
    std::cout << "Taking move (from available pockets on our side, L-to-R) = "
        << max_i << "\n";
    game.MakeMove(max_i);
    turn_i++;
  }

  std::cout << "**********************************************************\n";
  std::cout << "Outcome for A " << game.Outcome()[0] << "\n";
  std::cout << "Outcome for B " << game.Outcome()[1] << "\n";

  return 0;
}
