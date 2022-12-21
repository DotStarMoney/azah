#include <iostream>
#include <vector>

#include "games/tictactoe/tictactoe.h"
#include "games/tictactoe/tictactoe_network.h"
#include "mcts/network_dispatch_queue.h"
#include "mcts/playout_runner.h"
#include "nn/data_types.h"

namespace {

using Tictactoe = azah::games::tictactoe::Tictactoe;
using TictactoeNetwork = azah::games::tictactoe::TictactoeNetwork;
using TictactoeRunner = azah::mcts::PlayoutRunner<
    azah::games::tictactoe::Tictactoe,
    azah::games::tictactoe::TictactoeNetwork,
    1024,
    131072,
    4>;
using TictactoeRunnerConfig = azah::mcts::PlayoutConfig<
    azah::games::tictactoe::Tictactoe>;

}  // namespace

int main(int argc, char* argv[]) {
  azah::mcts::NetworkDispatchQueue<TictactoeNetwork> work_queue(4, 16384);
  
  TictactoeNetwork network;

  std::vector<azah::nn::DynamicMatrixRef> vars;
  network.GetVariables({}, vars);
  work_queue.SetAllVariables(vars);

  Tictactoe game;

  game.MakeMove(2);
  game.MakeMove(5);
  game.MakeMove(6);
  game.MakeMove(4);
  game.MakeMove(0);
  game.MakeMove(1);

  // X _ X
  // O _ O
  // O _ X

  TictactoeRunnerConfig config{
      .game = game,
      .n = 384,
      .outcome_weight = 0.50,
      .policy_weight = 0.33, 
      .revisit_weight = 0.25,
      .policy_noise = 0.08};

  TictactoeRunner runner;

  auto result = runner.Playout(config, work_queue);

  std::cout << "Odds, with the player to move in the first row:\n" << result.outcome << std::endl;
  std::cout << "Player to move's policy:\n" << result.policy << std::endl;
  std::cout << "Player to move's best move:\n" << result.max_option_i << std::endl;

  return 0;
}
