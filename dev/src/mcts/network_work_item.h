#ifndef AZAH_MCTS_NETWORK_WORK_ITEM_H_
#define AZAH_MCTS_NETWORK_WORK_ITEM_H_

namespace azah {
namespace mcts {

template <typename GameNetworkSubclass>
class GameNetworkWorkItem {
 public:
  virtual void operator()(GameNetworkSubclass* local_network) const = 0;
};

}  // namespace mcts
}  // namespace azah

#endif  // AZAH_MCTS_NETWORK_WORK_ITEM_H_
