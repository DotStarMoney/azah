#include <iostream>
#include <vector>

#include "games/tictactoe/tictactoe.h"
#include "games/tictactoe/tictactoe_network.h"
#include "glog/logging.h"
#include "mcts/self_player.h"

namespace {

using TictactoeSelfPlayer = azah::mcts::SelfPlayer<
    azah::games::tictactoe::Tictactoe,
    azah::games::tictactoe::TictactoeNetwork,
    1024,
    131072,
    4,
    16384>;

}  // namespace

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);

  TictactoeSelfPlayer self_player(4);
  azah::mcts::SelfPlayConfig config{
      .playouts_n = 384,
      .learning_rate = 0.01,
      .outcome_weight = 0.50,
      .policy_weight = 0.33,
      .revisit_weight = 0.25,
      .policy_noise = 0.08};

  int steps_n = 0;
  for (int i = 0; i < 100; ++i) {
    auto loss = self_player.Train(5, config);
    steps_n += 5;
    std::cout << "Step: " << steps_n << ", " << loss << std::endl;
  }
 
  return 0;
}
