#include <chrono>
#include <iostream>
#include <tuple>
#include <thread>

#include "mcts/lock_by_key.h"

int main(int argc, char* argv[]) {
  azah::mcts::LockByKey<std::tuple<int, int>, 32> locks;

  std::thread t1([&locks]() {
        std::cout << "Thread stalling for 2s." << "\n";
        auto lock1 = locks.Lock({1, 2});
        std::cout << "Thread got lock." << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        std::cout << "Thread done." << "\n";
      });

  {
    std::cout << "Main stalling for 2s." << "\n";
    auto lock1 = locks.Lock({2, 3});
    std::cout << "Main got lock." << "\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    std::cout << "Main done." << "\n";
  }

  t1.join();

  std::cout << "All done.";

  return 0;
}
