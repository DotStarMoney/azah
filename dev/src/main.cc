#include <iostream>
#include <vector>

#include "games/tictactoe/tictactoe.h"
#include "games/tictactoe/tictactoe_network.h"
#include "glog/logging.h"
#include "mcts/self_player.h"
#include "nn/data_types.h"

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
      .playouts_n = 128,
      .learning_rate = 0.01,
      .outcome_weight = 0.33,
      .policy_weight = 0.33,
      .revisit_weight = 0.25,
      .policy_noise = 0.08};
   
  int steps_n = 0;
  for (int i = 0; i < 500; ++i) {
    auto loss = self_player.Train(5, config);
    steps_n += 5;
    std::cout << "Step: " << steps_n << ", " << loss << std::endl;
  }
 
  azah::games::tictactoe::Tictactoe bb;
  auto x = self_player.EvaluatePosition(bb, config);

  std::cout << x.best_move_option_i << ", " 
      << x.projected_outcome[0] << ", " 
      << x.projected_outcome[1] << std::endl;

  auto& network = self_player.get_network();
  
  network.SetConstants(network.input_constant_indices(),
                       bb.StateToMatrix());
  std::vector<azah::nn::DynamicMatrix> model_outputs;
  network.Outputs({0, 1}, model_outputs);

  std::cout << "Policy: " << std::endl;
  std::cout << model_outputs[0] << std::endl;

  std::cout << "Outcome: " << std::endl;
  std::cout << model_outputs[1] << std::endl;

  return 0;
}
