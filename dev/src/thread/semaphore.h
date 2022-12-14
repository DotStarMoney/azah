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

  void P();
  bool TryP();

  void V();

  void Drain();

 private:
  std::shared_mutex m_;
  std::condition_variable_any cv_;  // GUARDED_BY(m_)

  std::atomic_int32_t r_;
};

}  // namespace thread
}  // namespace azah

#endif  // AZAH_THREAD_SEMAPHORE_H_
