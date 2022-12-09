#include <iostream>
#include <tuple>

#include "mcts/state_cache.h"

int main(int argc, char* argv[]) {
  azah::mcts::StateCache<std::tuple<int, int>, float, 8, 4, 16> cache;

  float x[] = { 0.1, 0.2, 0.3, 0.4 };

  auto k0 = std::make_tuple<int, int>(1, 2);
  std::cout << cache.TryStore(k0, &(x[0]), 4) << "\n";

  auto k1 = std::make_tuple<int, int>(2, 3);
  std::cout << cache.TryStore(k1, &(x[1]), 3) << "\n";

  auto k2 = std::make_tuple<int, int>(3, 4);
  std::cout << cache.TryStore(k2, &(x[2]), 2) << "\n";

  auto k3 = std::make_tuple<int, int>(4, 5);
  std::cout << cache.TryStore(k3, &(x[0]), 4) << "\n";

  auto k4 = std::make_tuple<int, int>(5, 6);
  std::cout << cache.TryStore(k4, &(x[1]), 3) << "\n";

  auto k5 = std::make_tuple<int, int>(6, 7);
  std::cout << cache.TryStore(k5, &(x[2]), 2) << "\n";

  auto k6 = std::make_tuple<int, int>(7, 8);
  std::cout << cache.TryStore(k6, &(x[0]), 4) << "\n";


  std::cout << cache.TryLoad(k0, &(x[0]), 4) << "\n";
  std::cout << cache.TryLoad(k1, &(x[1]), 3) << "\n";
  std::cout << cache.TryLoad(k2, &(x[2]), 2) << "\n";
  std::cout << cache.TryLoad(k3, &(x[0]), 4) << "\n";
  std::cout << cache.TryLoad(k4, &(x[1]), 3) << "\n";
  std::cout << cache.TryLoad(k5, &(x[2]), 2) << "\n";
  std::cout << cache.TryLoad(k6, &(x[0]), 4) << "\n";


  return 0;
}
