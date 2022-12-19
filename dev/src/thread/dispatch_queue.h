#ifndef AZAH_THREAD_DISPATCH_QUEUE_H_
#define AZAH_THREAD_DISPATCH_QUEUE_H_

#include <stdint.h>

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "glog/logging.h"
#include "semaphore.h"

namespace azah {
namespace thread {

template <typename CallableWorkItem, typename ThreadState>
class DispatchQueue {
 public:
  DispatchQueue(const DispatchQueue&) = delete;
  DispatchQueue& operator=(const DispatchQueue&) = delete;
  
  DispatchQueue(uint32_t threads, uint32_t queue_length) :
      buffer_avail_(queue_length),
      buffer_elem_remain_(0),
      exit_(false),
      drain_(1),
      slot_(0),
      slot_working_(0),
      buffer_(queue_length),
      thread_state_(threads, nullptr) {
    if (queue_length <= 0) {
      LOG(FATAL) << "Queue length must be greater than 0.";
    }
    for (int i = 0; i < threads; ++i) {
      auto dispatch_fn = [this, i] {
            for (;;) {
              buffer_elem_remain_.P();
              if (exit_) return;
              WorkElement& work_element = buffer_[
                  slot_working_.fetch_add(1, std::memory_order_relaxed) 
                      % buffer_.size()];
              while (!work_element.ready.load(std::memory_order_relaxed)) {}
              (*(work_element.work))(this->thread_state_[i]);
              drain_.Inc();
              work_element.ready.exchange(false, std::memory_order_release);
              buffer_avail_.V();
            }
          };
      workers_.emplace_back(dispatch_fn);
    }
  }

  ~DispatchQueue() {
    exit_ = true;
    buffer_elem_remain_.Drain();
    for (auto& worker : workers_) {
      worker.join();
    }
  }

  // Thread safe.
  void AddWork(std::unique_ptr<CallableWorkItem> work) {
    drain_.Dec();
    buffer_avail_.P();
    WorkElement& work_element =
        buffer_[slot_.fetch_add(1, std::memory_order_relaxed) % buffer_.size()];
    work_element.work = std::move(work);
    work_element.ready.exchange(true, std::memory_order_acquire);
    buffer_elem_remain_.V();
  }

  // Thread safe.
  void Drain() {
    drain_.Wait();
  }

  // Not thread safe.
  void SetThreadState(uint32_t thread, ThreadState* state) {
    thread_state_[thread] = state;
  }

  uint32_t threads_n() const {
    return thread_state_.size();
  }

 private:
  Semaphore buffer_avail_;
  Semaphore buffer_elem_remain_;
  std::atomic_bool exit_;

  Semaphore drain_;

  std::atomic_uint32_t slot_;
  std::atomic_uint32_t slot_working_;

  struct WorkElement {
    WorkElement() : ready(false) {}
    std::unique_ptr<CallableWorkItem> work;
    std::atomic<bool> ready;
  };
  std::vector<WorkElement> buffer_;

  std::vector<std::thread> workers_;
  std::vector<ThreadState*> thread_state_;
};

}  // namespace thread
}  // namespace azah

#endif  // AZAH_THREAD_DISPATCH_QUEUE_H_
