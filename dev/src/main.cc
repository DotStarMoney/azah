#include <chrono>
#include <iostream>
#include <tuple>
#include <thread>

#include "mcts/lock_by_key.h"

int main(int argc, char* argv[]) {
  azah::mcts::LockByKey<std::tuple<int, int>, 32> locks;

  return 0;
}
