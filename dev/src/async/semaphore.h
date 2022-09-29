#ifndef AZAH_ASYNC_SEMAPHORE_H_
#define AZAH_ASYNC_SEMAPHORE_H_

#include <atomic>
#include <condition_variable>
#include <shared_mutex>

#include "absl/base/thread_annotations.h"
#include "util/noncopyable.h"

namespace azah {
namespace async {

// A bog-standard semaphore. The C++ standards committee thinks we're too stupid
// to have one, so here we are...
class Semaphore : public util::NonCopyable {
 public:
  Semaphore(int32_t init_resource);

  void P();
  bool TryP();

  void V();

  // If init_resource > 0, blocks until the internal resource counter is equal
  // to init_resource.
  void Join();

  void Drain();

 private:
  std::shared_mutex m_;
  std::shared_mutex join_m_;
  std::condition_variable_any cv_ GUARDED_BY(m_);
  std::condition_variable_any join_cv_ GUARDED_BY(join_m_);

  std::atomic_int32_t r_;

  const std::int32_t init_resource_;
};

}  // namespace async
}  // namespace azah

#endif  // AZAH_ASYNC_SEMAPHORE_H_
