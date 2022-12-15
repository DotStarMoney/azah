#ifndef AZAH_THREAD_SEMAPHORE_H_
#define AZAH_THREAD_SEMAPHORE_H_

#include <atomic>
#include <condition_variable>
#include <shared_mutex>

namespace azah {
namespace thread {

class Semaphore {
 public:
  Semaphore(const Semaphore&) = delete;
  Semaphore& operator=(const Semaphore&) = delete;

  Semaphore(int32_t init_resource);

  // Mode 1: limited resource synchronization
  //

  // Take one resource, wait if count is <= 0.
  void P();
  
  // Return one resource.
  void V();

  // Mode 2: wait for completion
  //
 
  // Wait for counter to be > 0.
  void Wait();

  // Increment counter.
  void Inc();

  // Decrement counter.
  void Dec();

  //
  //

  // Release all waiters, destroying the semaphore.
  void Drain();

 private:
  std::shared_mutex m_;
  std::condition_variable_any cv_;  // GUARDED_BY(m_)

  std::atomic_int32_t r_;
};

}  // namespace thread
}  // namespace azah

#endif  // AZAH_THREAD_SEMAPHORE_H_
