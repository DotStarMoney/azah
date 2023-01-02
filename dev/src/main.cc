#include <iostream>
#include <vector>

#include "games/game.h"
#include "games/tictactoe/tictactoe.h"
#include "games/tictactoe/tictactoe_network.h"
#include "glog/logging.h"
#include "mcts/rl_player.h"

namespace {

using Tictactoe = azah::games::tictactoe::Tictactoe;
using TictactoeNetwork = azah::games::tictactoe::TictactoeNetwork;
using RLPlayer = azah::mcts::RLPlayer<Tictactoe, TictactoeNetwork>;

}  // namespace


int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  
  RLPlayer player(2);

  RLPlayer::SelfPlayOptions options{
      .learning_rate = 0.01,
      .simulations_n = 128,
      .root_noise_alpha = 0.4,
      .root_noise_lerp = 0.25,
      .one_hot_breakover_moves_n = 2,
      .exploration_scale = 0.25};

  for (int i = 0; i < 200; ++i) {
    std::cout << "Playing 5 games..." << std::endl;
    auto losses = player.Train(5, options);
    std::cout << "Finished game " << i << ", with loss " << losses << std::endl;
  }

  std::cout << "FULL GAME -------------------------" << std::endl;
  Tictactoe game;
  
  bool x_move = true;
  while (game.State() == azah::games::GameState::kOngoing) {
    std::cout << "PLAYER " << (x_move ? "X" : "O") << "------------------------" 
        << std::endl;
    auto eval_results = player.Evaluate(game, options);

    int best_move_i = -1;
    float best_move_v = -1;

    std::cout << "Predicted moves:" << std::endl;
    for (int i = 0; i < game.CurrentMovesN(); ++i) {
      std::cout << eval_results.predicted_move[i] << std::endl;
      if (eval_results.predicted_move[i] > best_move_v) {
        best_move_v = eval_results.predicted_move[i];
        best_move_i = i;
      }
    }
    
    std::cout << "Predicted outcome:" << std::endl;
    for (int i = 0; i < 2; ++i) {
      std::cout << eval_results.predicted_outcome[i] << std::endl;
    }

    std::cout << "Taking move " << best_move_i << std::endl << std::endl;
    game.MakeMove(best_move_i);

    x_move = !x_move;
  }

  return 0;
}
