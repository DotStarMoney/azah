#include <iostream>
#include <vector>

#include "games/tictactoe/tictactoe.h"
#include "games/tictactoe/tictactoe_network.h"
#include "glog/logging.h"
#include "mcts/self_play.h"
#include "mcts/rl_player.h"
#include "nn/data_types.h"

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  
  azah::games::tictactoe::Tictactoe game;
  azah::games::tictactoe::TictactoeNetwork game_network;

  azah::mcts::self_play::Config config{
        .simulations_n = 128,
        .full_play = true,
        .root_noise_alpha = 0.4,
        .root_noise_lerp = 0.25,
        .one_hot_breakover_moves_n = 2,
        .exploration_scale = 1.0
      };

  auto results = azah::mcts::self_play::SelfPlay(config, game, &game_network);

  bool x_move = true;
  for (const auto& x : results) {
    if (x_move) {
      std::cout << "----------------X TO MOVE------------------\n";
    } else {
      std::cout << "----------------O TO MOVE------------------\n";
    }
    std::cout << "BOARD STATE\n";
    std::cout << x.state_inputs[0].reshaped(3, 3) << "\n";
    std::cout << "SEARCHED\n";
    std::cout << x.search_policy.reshaped(3, 3) << "\n";
    std::cout << "OUTCOME\n";
    std::cout << x.outcome << "\n";
    x_move = !x_move;
  }

  return 0;
}
