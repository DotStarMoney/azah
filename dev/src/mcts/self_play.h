#ifndef AZAH_MCTS_SELF_PLAY_H_
#define AZAH_MCTS_SELF_PLAY_H_

#include <math.h>
#include <stddef.h>

#include <algorithm>
#include <array>
#include <memory>
#include <random>
#include <vector>

#include "../games/game.h"
#include "../games/game_network.h"
#include "../nn/data_types.h"
#include "absl/container/flat_hash_map.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"
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
  float search_prob;

  // The number of times this edge has been traversed.
  int visits_n;

  // The source index of the parent node.
  const std::size_t parent_i;

  // The source index of the child node. If -1, there is no child node at the
  // other end, and this node wi
  std::size_t child_i;

  bool barren() const {
    return child_i == -1;
  }
};

template <typename Game>
struct TreeNode {
  TreeNode(Game&& game, std::size_t parent_i) :
      game(std::move(game)), parent_i(parent_i) {
    visit_sum = 0;
  }

  // The game state at this node. 
  const Game game;

  // The indices of the child edges for this move. These follow the same order
  // as the moves in Game::MakeMove.
  std::vector<std::size_t> children_i;

  // The index of the parent edge. If -1, this is the root.
  std::size_t parent_i;
  
  // The outcome predicted at this game state.
  std::array<float, Game::players_n()> predicted_outcome;

  // The sum of the visit counts (visits_n) of each edge.
  int visit_sum;

  bool root() const {
    return parent_i == -1;
  }

  bool terminal() const {
    return children_i.empty();
  }
};

template <typename Game, typename GameNetwork>
class GameTree {
 public:
  std::vector<TreeNode<Game>> nodes;
  std::vector<TreeEdge<Game>> edges;

  void Search(TreeNode<Game>* node, GameNetwork* network, 
              float exploration_scale, float root_noise_alpha, 
              float root_noise_lerp, absl::BitGenRef bitgen) {

    // Step 1 is to explore the tree and find a leaf outcome to propagate up.
    std::array<float, Game::players_n()> leaf_outcome;
    for (;;) {
      // Selecting an edge *at* the root is a little more involved since we have
      // to factor in some exploration noise.
      std::unique_ptr<float[]> noise;
      if (node->root()) {
        noise = GameTree<Game, GameNetwork>::DirichletNoise(
            root_noise_alpha, node->children_i.size(), bitgen);
      }
      float max_value = -1.0f;
      TreeEdge<Game>* max_edge;
      int max_edge_index;
      for (int child_i = 0; child_i < node->children_i.size(); ++child_i) {
        TreeEdge<Game>& edge = edges[node->children_i[child_i]];
        float policy_value = node->root()
            ? (noise[child_i] - edge.search_prob) * root_noise_lerp
                + edge.search_prob
            : edge.search_prob; 
        // We consider the outcome to be the projection from whoever's turn it
        // is, giving us our min-max like behavior.
        float edge_value = edge.outcome[node->game.CurrentPlayerI()]
            + exploration_scale * policy_value * (
                std::sqrtf(static_cast<float>(node->visit_sum))
                    / static_cast<float>(1 + edge.visits_n));
        if (edge_value > max_value) {
          max_value = edge_value;

          max_edge = &edge;
          max_edge_index = child_i;
        }  
      }

      // Now either traverse the edge, expand it, or just pass up its value up
      // if it's terminal.
      if (!max_edge->barren()) {
        node = &(nodes[max_edge->child_i]);
        if (node->terminal()) {
          leaf_outcome = max_edge->outcome;
          break;
        }
      } else {
        Game expanded_game(node->game);
        expanded_game.MakeMove(max_edge_index);
        leaf_outcome = ExpandNode(std::move(expanded_game), 
                                  node->children_i[max_edge_index], network);
        node = &(nodes.back());
        break;
      }
    }

    // Step 2 is to propagate up leaf_outcome from node.
    while (!node->root()) {
      TreeEdge<Game>& parent_edge = edges[node->parent_i];
      ++(parent_edge.visits_n);
      for (int i = 0; i < Game::players_n(); ++i) {
        parent_edge.acc_outcome[i] += leaf_outcome[i];
        parent_edge.outcome[i] = parent_edge.acc_outcome[i] 
            / static_cast<float>(parent_edge.visits_n);
      }
      node = &(nodes[parent_edge.parent_i]);
      ++(node->visit_sum);
    }
  }

  const std::array<float, Game::players_n()>& ExpandNode(
      Game&& expanded_game, std::size_t source_edge_i, GameNetwork* network) {
    nodes.emplace_back(std::move(expanded_game), source_edge_i);
    TreeNode<Game>& node = nodes.back();
    std::size_t node_i = nodes.size() - 1;

    if (!edges.empty()) {
      edges[source_edge_i].child_i = node_i;
      node.parent_i = source_edge_i;
    } else {
      node.parent_i = -1;
    }

    if (node.game.State() == games::GameState::kOver) {
      node.predicted_outcome = node.game.Outcome();
      return node.predicted_outcome;
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

    // After creating the edges, we normalize the search probabilities because
    // we don't expect the model to.
    float policy_sum = 0.0f;
    for (int i = 0; i < node.game.CurrentMovesN(); ++i) {
      float policy = node.game.PolicyForMoveI(model_outputs[1], i);
      policy_sum += policy;
      edges.emplace_back(policy, node_i);
      node.children_i.push_back(edges.size() - 1);
    }
    for (auto& edge_i : node.children_i) {
      edges[edge_i].search_prob /= policy_sum;
    }

    // Since the model is trying to predict an outcome vector rotated s.t. the
    // current player is in the first row, un-rotate the prediction.
    for (int i = 0; i < Game::players_n(); ++i) {
      node.predicted_outcome[
          (i + node.game.CurrentPlayerI()) % Game::players_n()] = 
              model_outputs[0](i, 0);
    }

    return node.predicted_outcome;
  }
 
 private:
  static inline std::unique_ptr<float[]> DirichletNoise(
      float alpha, int n, absl::BitGenRef bitgen) {
    // 1D, so \beta = 1.
    std::gamma_distribution<float> gamma(alpha);
    auto noise = std::unique_ptr<float[]>(new float[n]);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
      noise[i] = gamma(bitgen);
      sum += noise[i];
    }
    for (int i = 0; i < n; ++i) {
      noise[i] /= sum;
    }
    return noise;
  }
};

static inline int SamplePolicy(const std::unique_ptr<float[]>& prop, int size, 
                               absl::BitGenRef bitgen) {
  int move_index = 0;

  float uniform = absl::Uniform(bitgen, 0.0f, 1.0f);
  float total = 0.0f;
  float prev_total = total;
  for (; move_index < size; ++move_index) {
    total += prop[move_index];
    if ((uniform >= prev_total) && (uniform < total)) break;
    prev_total = total;
  }
  // Handle round-off errors.
  return (move_index == size) ? size - 1 : move_index;
}

}  // namespace internal

template <typename Game>
struct MoveOutcome {
  // The outcome order-rotated s.t. the player who's move it is to make is in
  // the first row.
  //
  // If not playing a full game, the value of this field is unspecified.
  nn::Matrix<Game::players_n(), 1> outcome;

  // The visit proportions across move options at the searched game state.
  nn::DynamicMatrix search_policy;

  // The index of the policy head of the games::GameNetwork that search_policy
  // applies to.
  std::size_t search_policy_class_i;

  // The network inputs that should yield outcome and search_policy.
  std::vector<nn::DynamicMatrix> state_inputs;

  // The move index with the highest search value, a.k.a "the move to make."
  // This is not necessarily the move taken during self-play since that is
  // chosen by sampling the policy vector.
  std::size_t max_option_i;
};

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
  // valued move. If !full_play, the root search policy is never converted to
  // one-hot.
  //
  // AlphaZero uses 30.
  int one_hot_breakover_moves_n;

  // A multiplier on the upper-confidence-bound that encourages exploration when
  // higher.
  float exploration_scale;
};

template <typename Game, typename GameNetwork>
std::vector<MoveOutcome<Game>> SelfPlay(const Config& config, const Game& game,
                                        GameNetwork* network) {
  if (game.State() == games::GameState::kOver) {
    LOG(FATAL) << "Self play cannot begin from a terminal state.";
  }
  internal::GameTree<Game, GameNetwork> tree;
  (void)tree.ExpandNode(Game(game), -1, network);
  std::size_t root_i = 0;
  internal::TreeNode<Game>* root = &(tree.nodes[root_i]);

  // These are parallel arrays.
  // 
  // We build out the results and current_player_i rows during self-play, and
  // when a winner is called, fill in the results outcome fields rotated to 
  // respect the current_player_i for that row.
  std::vector<MoveOutcome<Game>> results;
  std::vector<int> current_player_i;

  absl::BitGen bitgen;
  int total_moves = 0;
  while (root->game.State() == games::GameState::kOngoing) {
    // To make a move, we first grow the tree a bunch from this position.
    for (int sim_i = 0; sim_i < config.simulations_n; ++sim_i) {
      tree.Search(root, network, config.exploration_scale,
          config.root_noise_alpha, config.root_noise_lerp, bitgen);
      // The vector backing this pointer can change in Search.
      root = &(tree.nodes[root_i]);
    }
    const int moves_n = root->children_i.size();

    // Next, we take the search proportions at the root and create a policy
    // vector from them.
    auto search_policy = std::unique_ptr<float[]>(new float[moves_n]);
    float max_search_policy = 0.0f;
    int max_search_policy_i = -1;
    for (int move_i = 0; move_i < moves_n; ++move_i) {
      // Since we've been tracking node visit sums, this will come out
      // normalized.
      search_policy[move_i] = 
          static_cast<float>(tree.edges[root->children_i[move_i]].visits_n) 
              / static_cast<float>(root->visit_sum);
      if (search_policy[move_i] > max_search_policy) {
        max_search_policy = search_policy[move_i];
        max_search_policy_i = move_i;
      }
    }
    if (config.full_play && (total_moves >= config.one_hot_breakover_moves_n)) {
      for (int move_i = 0; move_i < moves_n; ++move_i) {
        search_policy[move_i] = (move_i == max_search_policy_i)
            ? 1.0f
            : 0.0f;
      }
    }

    // Next, copy the vectorized stats into the output.
    MoveOutcome<Game> move_outcome;
    move_outcome.search_policy_class_i = root->game.PolicyClassI();
    move_outcome.state_inputs = root->game.StateToMatrix();
    move_outcome.search_policy = root->game.PolicyMask();
    int move_i = 0;
    for (int policy_vec_i = 0; policy_vec_i < move_outcome.search_policy.rows();
        ++policy_vec_i) {
      if (move_outcome.search_policy(policy_vec_i, 0) == 0.0f) continue;
      move_outcome.search_policy(policy_vec_i, 0) = search_policy[move_i++];
    }
    move_outcome.max_option_i = max_search_policy_i;
    results.push_back(std::move(move_outcome));
    current_player_i.push_back(root->game.CurrentPlayerI());

    // If we're just looking at this one move, we can leave.
    if (!config.full_play) return std::move(results);

    // Next, and the last step in self-play, we sample from the search policy
    int move_index = internal::SamplePolicy(search_policy, 
                                            root->children_i.size(), bitgen);

    // Setting parent_i to -1 ensures future searches don't propagate anything
    // past the new root and that noise is added where appropriate.
    root_i = tree.edges[root->children_i[move_index]].child_i;
    root = &(tree.nodes[root_i]);
    root->parent_i = -1;

    ++total_moves;
  }

  // Finally, score the game, and copy the rotated outcome into the result rows. 
  auto outcome = root->game.Outcome();
  for (int i = 0; i < results.size(); ++i) {
    for (int player_i = 0; player_i < Game::players_n(); ++player_i) {
      results[i].outcome(
          (player_i + current_player_i[i]) % Game::players_n(), 0) = 
              outcome[player_i];
    }
  }

  return std::move(results);
}

}  // namespace self_play
}  // namespace mcts
}  // namespace azah

#endif  // AZAH_MCTS_SELF_PLAY_H_
 