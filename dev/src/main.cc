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
    256,
    131072,
    4>;
using TictactoeRunnerConfig = azah::mcts::PlayoutConfig<
    azah::games::tictactoe::Tictactoe>;

}  // namespace

int main(int argc, char* argv[]) {
  azah::mcts::NetworkDispatchQueue<TictactoeNetwork> work_queue(16, 128);
  
  TictactoeNetwork network;

  std::vector<azah::nn::DynamicMatrixRef> vars;
  network.GetVariables({}, vars);
  work_queue.SetAllVariables(vars);

  TictactoeRunnerConfig config{
      .game = Tictactoe(), 
      .n = 10, 
      .policy_weight = 0.5, 
      .revisit_weight = 0.2, 
      .policy_noise = 0.05};
  
  TictactoeRunner runner;

  auto result = runner.Playout(config, work_queue);

  return 0;
}
