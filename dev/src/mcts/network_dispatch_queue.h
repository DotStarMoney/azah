#ifndef AZAH_MCTS_NETWORK_DISPATCH_QUEUE_H_
#define AZAH_MCTS_NETWORK_DISPATCH_QUEUE_H_

#include <stdint.h>

#include <memory>
#include <vector>

#include "../nn/data_types.h"
#include "../thread/dispatch_queue.h"
#include "network_work_item.h"

namespace azah {
namespace mcts {

template <typename NetworkSubclass>
class NetworkDispatchQueue : 
    public thread::DispatchQueue<NetworkWorkItem<NetworkSubclass>, 
                                 NetworkSubclass> {
 public:
  NetworkDispatchQueue(const NetworkDispatchQueue&) = delete;
  NetworkDispatchQueue& operator=(const NetworkDispatchQueue&) = delete;

  NetworkDispatchQueue(uint32_t threads, uint32_t queue_length) : 
      thread::DispatchQueue<CallableWorkItem, NetworkSubclass>(
          uint32_t threads, uint32_t queue_length) {
    for (int i = 0; i < threads_n(); ++i) {
      networks_.push_back(std::make_unique<NetworkSubclass>());
      SetThreadState(i, networks_.back().get());
    }
  }

  void SetAllConstants(const std::vector<uint32_t>& variables_i,
                       const std::vector<nn::DynamicMatrix>& variables) {
    for (auto& network_ptr : networks_) {
      network_ptr->SetVariables(variables_i, variables);
    }
  }

  void SetAllVariables(const std::vector<uint32_t>& constants_i,
                       const std::vector<nn::DynamicMatrix>& constants) {
    for (auto& network_ptr : networks_) {
      network_ptr->SetConstants(constants_i, constants);
    }
  }

 private:
  std::vector<std::unique_ptr<NetworkSubclass>> networks_;

};

}  // namespace mcts
}  // namespace azah

#endif  // AZAH_MCTS_NETWORK_DISPATCH_QUEUE_H_
