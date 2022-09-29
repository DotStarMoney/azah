#include "semaphore.h"

#include <atomic>
#include <condition_variable>
#include <limits>
#include <shared_mutex>

#include "glog/logging.h"

namespace azah {
namespace async {

Semaphore::Semaphore(int32_t init_resource) 
    : r_(init_resource), init_resource_(init_resource) {
  CHECK_GE(init_resource, 0);
}

// Using an atomic resource count here allows us to avoid locking in the common
// case. Multiple threads can acquire resource simultaneously lock free. The
// only lock contention will occur when releasing resource and threads are
// waiting.
//
// If there is plenty of resource to go around however, no locking will take
// place.

void Semaphore::P() {
  std::shared_lock<std::shared_mutex> s_lock(m_);
  if (r_.fetch_add(-1, std::memory_order_acquire) <= 0) cv_.wait(s_lock);
}

bool Semaphore::TryP() {
  if (r_.fetch_add(-1, std::memory_order_acquire) <= 0) {
    std::unique_lock<std::shared_mutex> lock(m_);
    r_.fetch_add(1, std::memory_order_relaxed);
    cv_.notify_one();
    return false;
  }
  return true;
}

void Semaphore::V() {
  int32_t r = r_.fetch_add(1, std::memory_order_release);
  if (r < 0) {
    std::unique_lock<std::shared_mutex> lock(m_);
    cv_.notify_one();
  } else if ((init_resource_ > 0) && ((r + 1) == init_resource_)) {
    std::unique_lock<std::shared_mutex> lock(join_m_);
    join_cv_.notify_all();
  }
}

void Semaphore::Join() {
  std::shared_lock<std::shared_mutex> s_lock(join_m_);
  if (r_.load(std::memory_order_consume) != init_resource_) {
    join_cv_.wait(s_lock);
  }
}

// Ugh. Why are we still making this a macro??
#undef max

void Semaphore::Drain() {
  r_ = std::numeric_limits<int32_t>::max();
  cv_.notify_all();
}

}  // namespace async
}  // namespace azah
