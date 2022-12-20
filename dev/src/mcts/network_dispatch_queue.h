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

template <typename GameNetworkSubclass>
class NetworkDispatchQueue : 
    public thread::DispatchQueue<GameNetworkWorkItem<GameNetworkSubclass>,
                                 GameNetworkSubclass> {
 public:
  NetworkDispatchQueue(const NetworkDispatchQueue&) = delete;
  NetworkDispatchQueue& operator=(const NetworkDispatchQueue&) = delete;

  NetworkDispatchQueue(uint32_t threads, uint32_t queue_length) : 
      thread::DispatchQueue<GameNetworkWorkItem<GameNetworkSubclass>, 
                            GameNetworkSubclass>(threads, queue_length) {
    for (int i = 0; i < this->threads_n(); ++i) {
      networks_.push_back(std::make_unique<GameNetworkSubclass>());
      this->SetThreadState(i, networks_.back().get());
    }
  }

  void SetAllVariables(const std::vector<nn::DynamicMatrixRef>& variables) {   
    std::vector<uint32_t> range_i;
    for (uint32_t i = 0; i < variables.size(); range_i.push_back(i++));
    for (auto& network_ptr : networks_) {
      network_ptr->SetVariables(range_i, variables);
    }
  }

  void SetAlConstants(const std::vector<nn::DynamicMatrixRef>& constants) {
    std::vector<uint32_t> range_i;
    for (uint32_t i = 0; i < constants.size(); range_i.push_back(i++));
    for (auto& network_ptr : networks_) {
      network_ptr->SetVariables(range_i, constants);
    }
  }

 private:
  std::vector<std::unique_ptr<GameNetworkSubclass>> networks_;
};

}  // namespace mcts
}  // namespace azah

#endif  // AZAH_MCTS_NETWORK_DISPATCH_QUEUE_H_
