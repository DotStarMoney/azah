#include <iostream>
#include <vector>

#include "games/tictactoe/tictactoe.h"

// ADD WARM UP WITH STRAIGHT MCTS

int main(int argc, char* argv[]) {
  azah::games::tictactoe::Tictactoe game;
  
  std::cout << game.CurrentMovesN() << std::endl;

  // X, O, X
  // X, O, X
  // O, X, O

  game.MakeMove(0);
  game.MakeMove(0);
  game.MakeMove(0);

  float p[] = {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
  std::cout << game.PolicyToMoveI(std::span(p)) << std::endl;

  game.MakeMove(1);
  game.MakeMove(0);
  game.MakeMove(1);
  game.MakeMove(0);
  game.MakeMove(1);
  game.MakeMove(0);

  std::cout << static_cast<int>(game.State()) << std::endl;

  std::cout << game.Outcome()[0] << ", " << game.Outcome()[1] << std::endl;

  azah::games::tictactoe::Tictactoe game2;

  game2.MakeMove(0);
  game2.MakeMove(7);
  game2.MakeMove(0);
  game2.MakeMove(5);
  game2.MakeMove(0);

  std::cout << game2.Outcome()[0] << ", " << game2.Outcome()[1] << std::endl;

  azah::games::tictactoe::Tictactoe game3;

  game3.MakeMove(0);
  game3.MakeMove(7);
  game3.MakeMove(0);
  game3.MakeMove(5);
  game3.MakeMove(1);
  game3.MakeMove(3);

  std::cout << game3.Outcome()[0] << ", " << game3.Outcome()[1] << std::endl;

  std::vector<float> input(9, 5);
  game3.StateToVector(input);

  for (int i = 0; i < 9; ++i) {
    std::cout << input[i] << ", ";
  }
  std::cout << std::endl;

  return 0;
}
