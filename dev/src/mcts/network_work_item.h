#ifndef AZAH_MCTS_NETWORK_WORK_ITEM_H_
#define AZAH_MCTS_NETWORK_WORK_ITEM_H_

namespace azah {
namespace mcts {

template <typename NetworkSubclass>
class NetworkWorkItem {
 public:
  virtual void operator()(NetworkSubclass* local_network) const = 0;
};

}  // namespace mcts
}  // namespace azah

#endif  // AZAH_MCTS_NETWORK_WORK_ITEM_H_
