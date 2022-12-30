#ifndef AZAH_MCTS_WORK_QUEUE_H_
#define AZAH_MCTS_WORK_QUEUE_H_

#include <stdint.h>

#include "../thread/dispatch_queue.h"

namespace azah {
namespace mcts {
namespace internal {

class WorkQueueElement {
 public:
  void operator()(void* unused) {
    run();
  }
  virtual void run() = 0;
};

class WorkQueue : public thread::DispatchQueue<WorkQueueElement, void> {
 public:
  WorkQueue(const WorkQueue&) = delete;
  WorkQueue& operator=(const WorkQueue&) = delete;

  WorkQueue(uint32_t threads, uint32_t queue_length) :
      thread::DispatchQueue<WorkQueueElement, void>(threads, queue_length) {}

  void SetThreadState(uint32_t thread, void* state) = delete;
};

}  // namespace internal
}  // namespace mcts
}  // namespace azah

#endif  // AZAH_MCTS_WORK_QUEUE_H_
