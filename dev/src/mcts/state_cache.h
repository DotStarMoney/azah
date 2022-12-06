#ifndef AZAH_MCTS_STATE_CACHE_H_
#define AZAH_MCTS_STATE_CACHE_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <limits>
#include <mutex>
#include <tuple>
#include <vector>

#include "absl/hash/hash.h"
#include "glog/logging.h"

namespace azah {
namespace mcts {

template <typename HashKey, typename Value, int Blocks, int BlockDataBytes, 
          int RowsPerBlock>
class StateCache {
  static_assert(
      (BlockDataBytes % sizeof(Value) == 0) 
          && (BlockDataBytes >= sizeof(Value)), 
      "Size of Value must evenly divide BlockDataBytes");

 public:
  StateCache() : blocks_(new Block[Blocks]) {}
 
  class Key {
    friend class StateCache;
   public:
    Key(const HashKey& key) : key_ref_(key), hashed_key_(absl::HashOf(key)) {}
   private:
    const HashKey& key_ref_;
    std::size_t hashed_key_;
  };

  bool TryLoad(const Key& key, Value* dest, uint16_t dest_size_elements) {
    Block& block = KeyToBlock(key);
    std::unique_lock<std::mutex> lock(block.m);

    for (int row_i = 0; row_i < block.rows_n; ++row_i) {
      Row& row = block.rows[row_i];
      if ((row.hash != key.hashed_key_) || (row.key != key.key_ref_)) continue;
      if (row.data_size_elements != dest_size_elements) {
        LOG(FATAL) << "Cached data size elements does not match requested. " 
            << row.data_size_elements << " vs. " << dest_size_elements;
      }
      ++row.hit_count;
      std::memcpy(
          static_cast<void*>(dest), 
          static_cast<void*>(&(block.data[row.data_offset_elements])), 
          dest_size_elements * sizeof(Value));
      return true;
    }

    return false;
  }

  bool TryStore(const Key& key, const Value* src, uint16_t src_size_elements) {
    if (src_size_elements > (BlockDataBytes / sizeof(Value))) {
      LOG(FATAL) << "Store byte limit is " << BlockDataBytes;
    }
    Block& block = KeyToBlock(key);
    std::unique_lock<std::mutex> lock(block.m);


    // Check to see if there's a hole in the array that fits.
    // If there is, store ourselves there, leave.
    // If there would be if we defragged:
    //   defrag
    //   store ourselves in the hole, leave
    // If there isn't room at all, from biggest to smallest, make a list of what
    //   we would have to kick. Decrease the hit_count on those things. If all
    //   hit 0, overwrite. Otherweise, set hit counts on list to no smaller than
    //   1. Return false



  }

  void Clear() {
    for (int block_i = 0; block_i < Blocks; ++block_i) {
      blocks_.get()[block_i].rows_n = 0;
    }
  }

  static constexpr std::size_t cache_size() {
    return Blocks * sizeof(Block) + sizeof(StateCache);
  }

  static constexpr std::size_t block_size() {
    return sizeof(Block);
  }

 private:

  struct Row {
    std::size_t hash;
    HashKey key;
    uint32_t hit_count;

    uint16_t data_offset_elements;
    uint16_t data_size_elements;
  };

  struct Block {
    Value data[BlockDataBytes];
    Row rows[RowsPerBlock];

    uint16_t rows_n = 0;
    std::mutex m;
  };

  static constexpr std::size_t KeyToBlockHash(const Key& key) {
    return key.hashed_key_ % Blocks;
  }

  Block& KeyToBlock(const Key& key) {
    return blocks_.get()[KeyToBlockHash(key)];
  }

  std::unique_ptr<Block[]> blocks_;
};

}  // namespace mcts
}  // namespace azah

#endif  // AZAH_MCTS_STATE_CACHE_H_
