#ifndef AZAH_MCTS_VISIT_TABLE_H_
#define AZAH_MCTS_VISIT_TABLE_H_

#include "stdint.h"

#include <memory>
#include <mutex>
#include <shared_mutex>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"

namespace azah {
namespace mcts {

template <typename HashKey, int Shards = 1>
class VisitTable {
 public:
  VisitTable(const VisitTable&) = delete;
  VisitTable& operator=(const VisitTable&) = delete;

  VisitTable() : table_shards_(new TableShard[Shards]) {}

  // Thread safe.
  void Inc(const HashKey& key) {
    TableShard& shard = table_shards_[absl::HashOf(key) % Shards];
    std::unique_lock<std::shared_mutex> read_write_lock(shard.m);
    auto [iter, is_new] = shard.table.insert({key, 1});
    if (!is_new) ++(iter->second);
  }
 
  // Thread safe.
  int Get(const HashKey& key) {
    TableShard& shard = table_shards_[absl::HashOf(key) % Shards];
    std::shared_lock<std::shared_mutex> read_lock(shard.m);
    auto iter = shard.table.find(key);
    if (iter == shard.table.end()) return 0;
    return iter->second;
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
    std::shared_mutex m;
  };

  std::unique_ptr<TableShard[]> table_shards_;
};

}  // namespace mcts
}  // namespace azah

#endif  // AZAH_MCTS_VISIT_TABLE_H_
