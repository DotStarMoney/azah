#include <iostream>
#include <mutex>

#include "thread/dispatch_queue.h"

// ADD WARM UP WITH STRAIGHT MCTS

std::mutex m;

class State {
 public:
  int x;
};

class Work {
 public:
  Work(azah::thread::DispatchQueue<Work, State>* queue, bool one_more, int source) 
      : queue_(queue), one_more_(one_more), source_(source) {}

  void operator()(State* s) {
    if (one_more_) {
      {
        std::unique_lock<std::mutex> lock(m);
        std::cout << "One more in thread " << s->x << ", id=" << source_ 
            << std::endl;
      }
      queue_->AddWork(std::make_unique<Work>(queue_, false, source_));
    } else {
      {
        std::unique_lock<std::mutex> lock(m);
        std::cout << "Stopping in thread " << s->x << ", id=" << source_ 
            << std::endl;
      }
    }
  }
 
 private:
  azah::thread::DispatchQueue<Work, State>* queue_;
  const bool one_more_;
  const int source_;
};

int main(int argc, char* argv[]) {
  azah::thread::DispatchQueue<Work, State> queue(4, 16);

  std::cout << "Starting..." << std::endl;

  State local_state[] = {0, 1, 2, 3};

  queue.SetThreadState(0, &(local_state[0]));
  queue.SetThreadState(1, &(local_state[1]));
  queue.SetThreadState(2, &(local_state[2]));
  queue.SetThreadState(3, &(local_state[3]));

  queue.AddWork(std::make_unique<Work>(&queue, true, 1));
  queue.AddWork(std::make_unique<Work>(&queue, true, 2));
  queue.AddWork(std::make_unique<Work>(&queue, true, 3));
  queue.AddWork(std::make_unique<Work>(&queue, true, 4));
  queue.AddWork(std::make_unique<Work>(&queue, true, 5));

  queue.Drain();

  std::cout << "NEXT" << std::endl;

  queue.AddWork(std::make_unique<Work>(&queue, true, 1));
  queue.AddWork(std::make_unique<Work>(&queue, true, 2));
  queue.AddWork(std::make_unique<Work>(&queue, true, 3));
  queue.AddWork(std::make_unique<Work>(&queue, true, 4));
  queue.AddWork(std::make_unique<Work>(&queue, true, 5));

  queue.Drain();

  return 0;
}
