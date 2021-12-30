#include "async/work_queue.h"

#include <functional>
#include <thread>

#include "glog/logging.h"

namespace azah {
namespace async {

WorkQueue::WorkQueue(uint32_t workers_n, uint32_t queue_length) : 
    buffer_avail_(queue_length),
    buffer_elem_remain_(0),
    exit_(false),
    slot_(0),
    slot_unclaimed_(0),
    buffer_(queue_length) {
  for (uint32_t i = 0; i < workers_n; ++i) {
    workers_.emplace_back(new std::thread([this, i] {
        for (;;) {
          buffer_elem_remain_.P();
          if (exit_) return;
          WorkElement& work_element = buffer_[
              slot_unclaimed_.fetch_add(1, std::memory_order_relaxed) 
                  % buffer_.size()];
          while (!work_element.ready.load(std::memory_order_consume)) {}
          work_element.work(i);
          work_element.ready.exchange(false, std::memory_order_release);
          buffer_avail_.V();
        }
      }));
  }
}

WorkQueue::~WorkQueue() {
  exit_ = true;
  buffer_elem_remain_.Drain();
  for (const auto& worker : workers_) {
    worker->join();
  }
}

void WorkQueue::AddWork(WorkItem f) {
  buffer_avail_.P();
  AddWorkInternal(f);
}

bool WorkQueue::TryAddWork(WorkItem f) {
  if (!buffer_avail_.TryP()) return false;
  AddWorkInternal(f);
  return true;
}

void WorkQueue::AddWorkInternal(WorkItem f) {
  WorkElement& work_element =
    buffer_[slot_.fetch_add(1, std::memory_order_relaxed) % buffer_.size()];
  work_element.work = f;
  work_element.ready.exchange(true, std::memory_order_acquire);
  buffer_elem_remain_.V();
}

void WorkQueue::Join() {
  //
  // TODO!!!!!
  //
  // If empty, leave
  // When a thread V's, it reads if there...
}

}  // namespace async
}  // namespace azah
