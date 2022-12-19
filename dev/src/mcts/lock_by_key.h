#ifndef AZAH_MCTS_LOCK_BY_KEY_H_
#define AZAH_MCTS_LOCK_BY_KEY_H_

#include <memory>
#include <mutex>

#include "absl/hash/hash.h"

namespace azah {
namespace mcts {

template <typename HashKey, int Shards = 1>
class LockByKey {
 public:
  LockByKey(const LockByKey&) = delete;
  LockByKey& operator=(const LockByKey&) = delete;

  LockByKey() : lock_shards_(new std::mutex[Shards]) {}

  // Thread safe.
  std::unique_lock<std::mutex> Lock(const HashKey& key) {
    return std::unique_lock<std::mutex>(
        lock_shards_[absl::HashOf(key) % Shards]);
  }

 private:
  std::unique_ptr<std::mutex[]> lock_shards_;
};

}  // namespace mcts
}  // namespace azah

#endif  // AZAH_MCTS_LOCK_BY_KEY_H_
