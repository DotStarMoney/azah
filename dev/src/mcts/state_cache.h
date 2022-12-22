#ifndef AZAH_MCTS_STATE_CACHE_H_
#define AZAH_MCTS_STATE_CACHE_H_

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>

#include "absl/hash/hash.h"
#include "glog/logging.h"

namespace azah {
namespace mcts {
namespace internal {

template <typename HashKey, typename Value, int Blocks, int RowsPerBlock,
          int ValuesPerRow>
class StateCache {
  static_assert(ValuesPerRow <= 65536, 
                "ValuesPerRow cannot exceed uint16 max.");

 public:
  StateCache(const StateCache&) = delete;
  StateCache& operator=(const StateCache&) = delete;

  StateCache() : blocks_(new Block[Blocks]) {
    Clear();
  }

  class TempKey {
    friend class StateCache;
   public:
    TempKey(const HashKey& key) : 
        key_ref_(key), hashed_key_(absl::HashOf(key)) {}
   private:
    const HashKey& key_ref_;
    std::size_t hashed_key_;
  };

  // Thread safe.
  bool TryLoad(const TempKey& key, Value* dest, int dest_size_elements) {
    Block& block = KeyToBlock(key);
    std::shared_lock<std::shared_mutex> read_lock(block.m);

    for (int row_i = 0; row_i < block.rows_n; ++row_i) {
      Row& row = block.rows[row_i];
      if ((row.hash != key.hashed_key_) || (row.key != key.key_ref_)) continue;
      if (row.elements_n != dest_size_elements) {
        LOG(FATAL) << "Cached data size elements does not match requested. " 
            << row.elements_n << " vs. " << dest_size_elements;
      }
      row.hit_count.fetch_add(1, std::memory_order_relaxed);
      std::memcpy(
          static_cast<void*>(dest), 
          static_cast<const void*>(&(block.data[row_i * ValuesPerRow])),
          dest_size_elements * sizeof(Value));
      return true;
    }

    return false;
  }

  // Thread safe.
  bool TryStore(const TempKey& key, const Value* src, int src_size_elements) {
    if (src_size_elements > ValuesPerRow) {
      LOG(FATAL) << "Store element limit is " << ValuesPerRow;
    }
    Block& block = KeyToBlock(key);
    std::unique_lock<std::shared_mutex> read_write_lock(block.m);

    int dest_row_i;
    if (block.rows_n < RowsPerBlock) {
      dest_row_i = block.rows_n++;
    } else {
      int i_preference = (key.hashed_key_ ^ kPreferenceSalt) % RowsPerBlock;

      int fewest_hit_n = block.rows[0].hit_count.load(
          std::memory_order_relaxed);
      int fewest_hit_i = 0;
      int fewest_hit_prox = i_preference;

      for (int row_i = 1; row_i < RowsPerBlock; ++row_i) {
        Row& row = block.rows[row_i];
        
        int proximity = std::abs(row_i - i_preference);
        uint32_t hit_count = row.hit_count.load(std::memory_order_relaxed);

        if ((hit_count < fewest_hit_n)
            || ((hit_count == fewest_hit_n) && (proximity < fewest_hit_prox))) {
          fewest_hit_n = hit_count;
          fewest_hit_i = row_i;
          fewest_hit_prox = proximity;
        }
      }

      if (fewest_hit_n > 1) {
        block.rows[fewest_hit_i].hit_count.store(fewest_hit_n - 1, 
                                                 std::memory_order_relaxed);
        return false;
      }

      dest_row_i = fewest_hit_i;
    }
    Row& dest_row = block.rows[dest_row_i];

    dest_row.key = key.key_ref_;
    dest_row.hash = key.hashed_key_;
    dest_row.hit_count.store(1, std::memory_order_relaxed);
    dest_row.elements_n = src_size_elements;

    std::memcpy(
        static_cast<void*>(&(block.data[dest_row_i * ValuesPerRow])),
        static_cast<const void*>(src),
        src_size_elements * sizeof(Value));

    return true;
  }

  // Not thread safe.
  void Clear() {
    for (int block_i = 0; block_i < Blocks; ++block_i) {
      blocks_[block_i].rows_n = 0;
    }
  }

  static constexpr int values_per_row_n() { return ValuesPerRow; }

 private:
  struct Row {
    HashKey key;
    std::size_t hash;
    std::atomic<uint32_t> hit_count;
    uint16_t elements_n;
  };

  struct Block {
    Value data[RowsPerBlock * ValuesPerRow];
    Row rows[RowsPerBlock];
    uint16_t rows_n;
    std::shared_mutex m;
  };

  static inline std::size_t KeyToBlockHash(const TempKey& key) {
    return key.hashed_key_ % Blocks;
  }

  Block& KeyToBlock(const TempKey& key) const {
    return blocks_[KeyToBlockHash(key)];
  }

  static constexpr std::size_t kPreferenceSalt = 0xED68495EE063E1D9;

  std::unique_ptr<Block[]> blocks_;
};

}  // namespace internal
}  // namespace mcts
}  // namespace azah

#endif  // AZAH_MCTS_STATE_CACHE_H_
