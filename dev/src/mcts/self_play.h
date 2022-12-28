#ifndef AZAH_MCTS_SELF_PLAY_H_
#define AZAH_MCTS_SELF_PLAY_H_

#include <stddef.h>

#include <algorithm>
#include <array>
#include <vector>

#include "../games/game.h"
#include "../games/game_network.h"
#include "../nn/data_types.h"
#include "absl/container/flat_hash_map.h"
#include "glog/logging.h"

namespace azah {
namespace mcts {
namespace self_play {
namespace internal {

template <typename Game>
struct TreeEdge {
  TreeEdge(float search_prob, std::size_t parent_i) :
      search_prob(search_prob), parent_i(parent_i) {
    std::fill(outcome.begin(), outcome.end(), 0.0f);
    std::fill(acc_outcome.begin(), acc_outcome.end(), 0.0f);
    visits_n = 0;
    child_i = -1;
  }

  // The sum of all outcomes predicted or discovered through this edge.
  std::array<float, Game::players_n()> acc_outcome;

  // outcome = acc_outcome / visits_n.
  std::array<float, Game::players_n()> outcome;

  // The predicted search probability along this edge.
  const float search_prob;

  // The number of times this edge has been traversed.
  std::size_t visits_n;

  // The source index of the parent node.
  const std::size_t parent_i;

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
  std::vector<std::size_t> children_i;

  // The index of the parent edge. If -1, this is the root.
  std::size_t parent_i;
};

template <typename Game, typename GameNetwork>
struct GameTree {
  std::vector<TreeNode> nodes;
  std::vector<TreeEdge> edges;

  std::array<float, Game::players_n()> ExpandNode(
      Game&& expanded_game, std::size_t source_edge_i, GameNetwork* network) {
    nodes.emplace_back(std::move(expanded_game), {}, {source_edge_i});
    TreeNode& node = nodes.back();
    std::size_t node_i = node_i;

    if (!edges.empty()) {
      edges[source_edge_i].child_i = node_i;
      node.parent_i = source_edge_i;
    } else {
      node.parent_i = -1;
    }

    if (node.game.State() == games::GameState::kOver) {
      return node.game.Outcome();
    } 

    network->SetConstants(network->input_constant_indices(), 
                          node.game.StateToMatrix());
    std::vector<nn::DynamicMatrix> model_outputs;
    network->Outputs(
        {
            network->outcome_output_index(),
            network->policy_output_indices()[node.game.PolicyClassI()]
        },
        model_outputs);
    for (int i = 0; i < node.game.CurrentMovesN(); ++i) {
      edges.emplace_back(node.game.PolicyForMoveI(model_outputs[1], i), node_i);
      node.children.push_back(edges.size() - 1);
    }

    std::array<float, Game::players_n()> projected_outcome;
    for (int i = 0; i < Game::players_n(); ++i) {
      projected_outcome[(i + node.game.CurrentPlayerI()) % Game::players_n()] =
          model_outputs[0](i, 0);
    }

    return projected_outcome;
  }
};

}  // namespace internal

template <typename Game>
struct GameOutcome {
  // The outcome order-rotated s.t. the player who's move it is to make is in
  // the first row.
  nn::Matrix<Game::players_n(), 1> outcome;

  // The visit proportions across move options at the searched game state.
  nn::DynamicMatrix search_policy;

  // The index of the policy head of the games::GameNetwork that search_policy
  // applies to.
  std::size_t search_policy_class_i;

  // The network inputs that should yield outcome and search_policy.
  std::vector<nn::DynamicMatrix> state_inputs;
};

enum class TerminationCriteria {
  kUnknown = 0,

  // The game ended because the resignation threshold was exceeded.
  kResign = 1,

  // The game ended because the game was over.
  kComplete = 2
};

template <typename Game>
struct Result {
  // The rows corresponding to the actual move search policies and the final
  // outcome as it stood for each player (redundantly included with each row
  // so we don't have to figure out how to rotate the outcome vector later). 
  std::vector<GameOutcome> rows;

  // How the game ultimately ended.
  TerminationCriteria termination;
};

template <typename Game, typename GameNetwork>
struct Config {
  // The total number of MCTS simulations to perform per-move.
  //
  // AlphaZero uses 800.
  int simulations_n;

  // If true, the game is played to completion from the provided game state.
  // Otherwise, only the result of searching a single move is returned.
  bool full_play;

  // The alpha value of Dirichlet noise added to the root search policy. Noise
  // is generated at every visit to the root.
  //
  // AlphaZero uses 0.3 for chess.
  float root_noise_alpha;

  // The amount interpolated between the predicted root policy and root noise.
  //
  // AlphaZero uses 0.25.
  float root_noise_lerp;

  // One-hot policy breakover threshold: the number of moves after which the
  // search policies returned from self-play are just one-hot of the maximum
  // valued move.
  //
  // AlphaZero uses 30.
  int one_hot_breakover_moves_n;

  // An outcome threshold for a player below which the player will resign. The
  // resignation critera is that for a given game state, both the root outcome
  // and the maximum child outcome after MCTS is below this threshold.
  float resignation_ubound;

  // A multiplier on the upper-confidence-bound that encourages exploration when
  // higher.
  float exploration_scale;
};

template <typename Game, typename GameNetwork>
Result<Game> SelfPlay(const Config<Game, GameNetwork>& config, 
                      const Game& game,
                      GameNetwork* network) {
  if (config.game.State() == games::GameState::kOver) {
    LOG(FATAL) << "Self play cannot begin from a terminal game state.";
  }
  internal::GameTree<Game, GameNetwork> tree;
  float root_value = tree.ExpandNode(game, -1, network);
  std::size_t root = 0;

  while (tree.nodes[root].game.State() == games::GameState::kOngoing) {
    
    //
    // Start a simulation (copy a game, etc...)
    //

    for (int playout_i = 0; playout_i < config.simulations_n; ++playout_i) {

    }

    // Check for resign.
  }
}

}  // namespace self_play
}  // namespace mcts
}  // namespace azah

#endif  // AZAH_MCTS_SELF_PLAY_H_
 