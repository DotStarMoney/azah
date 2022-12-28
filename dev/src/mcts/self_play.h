#ifndef AZAH_MCTS_SELF_PLAY_H_
#define AZAH_MCTS_SELF_PLAY_H_

#include <stddef.h>

#include <array>
#include <type_traits>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "../games/game.h"
#include "../games/game_network.h"
#include "../nn/data_types.h"

namespace azah {
namespace mcts {
namespace internal {

template <typename Game>
struct TreeEdge {
  // The sum of all outcomes predicted or discovered through this edge.
  std::array<float, Game::players_n()> acc_outcome;

  // = acc_outcome / visits_n.
  std::array<float, Game::players_n()> outcome;

  // The predicted search probability along this edge.
  float search_prob;

  // The number of times this edge has been traversed.
  std::size_t visits_n;

  // The source index of the parent node.
  std::size_t parent_i;

  // The source index of the child node. If -1, there is no child node at the
  // other end.
  std::size_t child_i;
};

template <typename Game>
struct TreeNode {
  // The game state at this node. 
  const Game game;

  // The indices of the child edges for this move. These follow the same order
  // as the moves in Game::MakeMove.
  std::vector<std::size_t> children;

  // The indices of the parent nodes for this move. If empty, this is the
  // root.
  std::vector<std::size_t> parents;
};

}  // namespace internal

template <typename Game>
struct SelfPlayRow {
  // The outcome order-rotated s.t. the player who's move it is to make is in
  // the first row.
  nn::Matrix<Game::players_n(), 1> outcome;

  // The search proportion across predicted options.
  nn::DynamicMatrix search_policy;

  // The index of the policy head of the games::GameNetwork that search_policy
  // applies to.
  std::size_t search_policy_class_i;

  // The network inputs that should yield outcome and search_policy.
  std::vector<nn::DynamicMatrix> state_inputs;
};

enum class SelfPlayTerminationCriteria {
  kUnknown = 0,

  // The game ended because the resignation threshold was exceeded.
  kResign = 1,

  // The game ended because the game was over.
  kComplete = 2
};

template <typename Game>
struct SelfPlayResult {
  std::vector<SelfPlayRow> rows;
  SelfPlayTerminationCriteria termination;
};

template <typename Game, typename GameNetwork>
SelfPlayResult<Game> SelfPlay(/* ... */) {
  std::vector<TreeNode> nodes;
  std::vector<TreeEdge> edges;
  absl::flat_hash_map<std::string, std::size_t> uid_to_node;

  // TODO: Lots! One thing to consider is how to make this function generic
  //     enough s.t. it works in the case where we just want to know what move
  //     to make in the current state, rather than play a whole game.
  //

}

}  // namespace mcts
}  // namespace azah

#endif  // AZAH_MCTS_SELF_PLAY_H_
 