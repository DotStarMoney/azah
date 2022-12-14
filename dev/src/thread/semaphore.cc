#include "semaphore.h"

#include <stdint.h>

#include <atomic>
#include <condition_variable>
#include <limits>
#include <shared_mutex>

#include "glog/logging.h"

#undef max

namespace azah {
namespace thread {

Semaphore::Semaphore(int32_t init_resource) : r_(init_resource) {
  if (init_resource < 0) {
    LOG(FATAL) << "Resource must be greater than or equal to 0.";
  }
}

void Semaphore::P() {
  std::shared_lock<std::shared_mutex> s_lock(m_);
  if (r_.fetch_add(-1, std::memory_order_acquire) <= 0) cv_.wait(s_lock);
}

bool Semaphore::TryP() {
  if (r_.fetch_add(-1, std::memory_order_acquire) <= 0) {
    std::unique_lock<std::shared_mutex> lock(m_);
    r_.fetch_add(1, std::memory_order_release);
    cv_.notify_one();
    return false;
  }
  return true;
}

void Semaphore::V() {
  if (r_.fetch_add(1, std::memory_order_release) < 0) {
    std::unique_lock<std::shared_mutex> lock(m_);
    cv_.notify_one();
  }
}

void Semaphore::Drain() {
  std::unique_lock<std::shared_mutex> lock(m_);
  r_.store(std::numeric_limits<int32_t>::max(), std::memory_order_relaxed);
  cv_.notify_all();
}

}  // namespace thread
}  // namespace azah
