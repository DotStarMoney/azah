#include <iostream>

#include "games/tictactoe/tictactoe.h"

// ADD WARM UP WITH STRAIGHT MCTS

int main(int argc, char* argv[]) {
  azah::games::tictactoe::Tictactoe game;
  
  std::cout << game.CurrentMovesN() << std::endl;


  return 0;
}
