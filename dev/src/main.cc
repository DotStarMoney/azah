#include <iostream>

#include "games/tictactoe/tictactoe.h"
#include "games/tictactoe/tictactoe_network.h"
#include "mcts/playout_runner.h"

int main(int argc, char* argv[]) {
  azah::mcts::PlayoutRunner<
      azah::games::tictactoe::Tictactoe, 
      azah::games::tictactoe::TictactoeNetwork, 
      256, 
      131072, 
      4, 
      17> runner;


  return 0;
}
