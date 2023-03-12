#ifndef AZAH_MCTS_CALLBACKS_H_
#define AZAH_MCTS_CALLBACKS_H_

#include <stddef.h>

#include <type_traits>

namespace azah {
namespace mcts {

// Callbacks useful for monitoring training / evaluation progress.
class CallbacksBase {
 public:
  virtual void PreSearch(int replica_i) {}
  virtual void PostSearch(int replica_i, std::size_t current_moves_n) {}

  virtual void PreGame(int replica_i) {}
  virtual void PostGame(int replica_i, std::size_t total_moves_n) {}

  virtual void PreUpdate(int replica_i) {}
  virtual void PostUpdate(int replica_i, std::size_t total_updates_n) {}
};

template <typename T>
concept CallbacksType = std::is_base_of<CallbacksBase, T>::value;

// Internal callbacks passed to per-replica functions. You probably don't want
// this and instead want to extend CallbacksBase.
template <CallbacksType Callbacks>
class ReplicaCallbacks {
 public:
  ReplicaCallbacks(int replica_i, Callbacks& callbacks) :
      replica_i_(replica_i), callbacks_(callbacks) {}

  void PreSearch() {
    callbacks_.PreSearch(replica_i_);
  }

  void PostSearch(std::size_t current_moves_n) {
    callbacks_.PostSearch(replica_i_, current_moves_n);
  }

  void PreGame() {
    callbacks_.PreGame(replica_i_);
  }

  void PostGame(std::size_t total_moves_n) {
    callbacks_.PostGame(replica_i_, total_moves_n);
  }

  void PreUpdate() {
    callbacks_.PreUpdate(replica_i_);
  }

  void PostUpdate(std::size_t total_updates_n) {
    callbacks_.PostUpdate(replica_i_, total_updates_n);
  }

 private:
  const int replica_i_;
  Callbacks& callbacks_;
};

}  // namespace mcts
}  // namespace azah

#endif  // AZAH_MCTS_CALLBACKS_H_
