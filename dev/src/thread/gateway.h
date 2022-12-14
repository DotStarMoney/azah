#ifndef AZAH_THREAD_GATEWAY_H_
#define AZAH_THREAD_GATEWAY_H_

#include <atomic>
#include <condition_variable>
#include <shared_mutex>

namespace azah {
namespace thread {

class Gateway {
 public:
  Gateway(const Gateway&) = delete;
  Gateway& operator=(const Gateway&) = delete;

  Gateway();

  void Enter();

  void Unlock();

 private:
  std::atomic_bool blocking_;

  bool guarded_blocking_;  // GUARDED_BY(m_)

  std::shared_mutex m_;
  std::condition_variable_any cv_;  // GUARDED_BY(m_)
};
}  // namespace thread
}  // namespace azah

#endif  // AZAH_THREAD_GATEWAY_H_
