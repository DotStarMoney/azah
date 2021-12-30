#ifndef AZAH_ASYNC_WORKQUEUE_H_
#define AZAH_ASYNC_WORKQUEUE_H_

#include <atomic>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

#include "async/semaphore.h"
#include "util/noncopyable.h"

namespace azah {
namespace async {

// A multi-producer, multi-consumer work queue. Uses RAII to start the worker
// threads. Producers will block when calling AddWork while the queue is full.
// All references in added work must outlive the work itself. The queue will
// block during destruction until all work has completed.
//
// It is safe for workers to AddWork to the queue they are a part of, however
// this can lead to deadlock if too many threads are blocked on adding work to
// a full queue.
//
// All methods are thread safe.
class WorkQueue : public util::NonCopyable {
public:
  typedef std::function<void(uint32_t)> WorkItem;

  // Starts workers with the given queue length.
  WorkQueue(uint32_t workers_n = 2, uint32_t queue_length = 1);

  ~WorkQueue();

  // Adds work to the queue. This will block while the queue is full.
  //
  // Work items are a function that takes the worker number as a parameter. This
  // number is unique to each worker and on [0, workers_n].
  void AddWork(WorkItem f);

  // Adds work to the queue. Returns false iff the queue is full.
  //
  // Work items are a function that takes the worker number as a parameter. This
  // number is unique to each worker and on [0, workers_n].
  bool TryAddWork(WorkItem f);

  // Blocks until all work items are complete.
  void Join();

private:
  void AddWorkInternal(WorkItem f);

  Semaphore buffer_avail_;
  Semaphore buffer_elem_remain_;
  std::atomic_bool exit_;

  std::atomic_uint64_t slot_;
  std::atomic_uint64_t slot_unclaimed_;

  struct WorkElement {
    WorkElement() : ready(false) {}
    WorkItem work;
    std::atomic<bool> ready;
  };
  std::vector<WorkElement> buffer_;

  std::vector<std::unique_ptr<std::thread>> workers_;
};

}  // namespace async
}  // namespace azah

#endif  // AZAH_ASYNC_WORKQUEUE_H_
