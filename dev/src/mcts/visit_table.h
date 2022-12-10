#ifndef AZAH_MCTS_VISIT_TABLE_H_
#define AZAH_MCTS_VISIT_TABLE_H_

#include "stdint.h"
#include "stdlib.h"

#include <memory>
#include <mutex>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"

namespace azah {
namespace mcts {

template <typename HashKey, int Shards = 1>
class VisitTable {
 public:
  VisitTable() : table_shards_(new TableShard[Shards]) {}

  // Thread safe.
  int Inc(const HashKey& key) {
    TableShard& shard = table_shards_[absl::HashOf(key) % Shards];
    std::unique_lock<std::mutex> read_write_lock(shard.m);
    auto [iter, is_new] = shard.table.insert({key, 1});
    if (is_new) return 1;
    return ++(iter->second);
  }

  // Not thread safe.
  void Clear() {
    for (int i = 0; i < Shards; ++i) {
      table_shards_[i].table.clear();
    }
  }
 
 private:
  struct TableShard {
    absl::flat_hash_map<HashKey, uint32_t> table;
    std::mutex m;
  };

  std::unique_ptr<TableShard[]> table_shards_;
};

}  // namespace mcts
}  // namespace azah

#endif  // AZAH_MCTS_VISIT_TABLE_H_