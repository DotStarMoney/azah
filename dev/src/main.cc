#include <iostream>
#include <tuple>

#include "mcts/state_cache.h"
#include "mcts/visit_table.h"

int main(int argc, char* argv[]) {
  azah::mcts::VisitTable<std::tuple<int, int>, 3> table;

  std::cout << table.FetchInc({1, 2}) << "\n";
  std::cout << table.FetchInc({2, 3}) << "\n";
  std::cout << table.FetchInc({3, 4}) << "\n";
  std::cout << table.FetchInc({4, 5}) << "\n";
  std::cout << table.FetchInc({5, 6}) << "\n";

  std::cout << table.FetchInc({1, 2}) << "\n";
  std::cout << table.FetchInc({2, 3}) << "\n";
  std::cout << table.FetchInc({3, 4}) << "\n";
  std::cout << table.FetchInc({4, 5}) << "\n";
  std::cout << table.FetchInc({5, 6}) << "\n";

  table.Clear();

  std::cout << table.FetchInc({1, 2}) << "\n";
  std::cout << table.FetchInc({2, 3}) << "\n";
  std::cout << table.FetchInc({3, 4}) << "\n";
  std::cout << table.FetchInc({4, 5}) << "\n";
  std::cout << table.FetchInc({5, 6}) << "\n";

  std::cout << table.FetchInc({1, 2}) << "\n";
  std::cout << table.FetchInc({2, 3}) << "\n";
  std::cout << table.FetchInc({3, 4}) << "\n";
  std::cout << table.FetchInc({4, 5}) << "\n";
  std::cout << table.FetchInc({5, 6}) << "\n";

  return 0;
}
