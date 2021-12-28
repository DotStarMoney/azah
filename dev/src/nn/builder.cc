#include "builder.h"

#include <memory>
#include <string_view>

#include "glog/logging.h"

namespace azah {
namespace nn {

using std::string_view;

void Builder::CheckId(NodeId id) const {
  CHECK_LT(id, nodes_.size());
  CHECK_NE(nodes_[id]->type, Builder::Type::kOutput);
}

Builder::NodeId Builder::Input(string_view input_name, int channel_n) {
  nodes_.emplace_back(new InputNode(input_name, channel_n));
  auto id = nodes_.size() - 1;
  inputs_.push_back(id);
  return id;
}

void Builder::Output(Builder::NodeId in, string_view output_name) {
  CheckId(in);
  nodes_.emplace_back(new OutputNode(in, output_name));
}

Builder::NodeId Builder::Dense(Builder::NodeId in, int channel_n, bool add_bias,
                               int group_n) {
  CheckId(in);
  nodes_.emplace_back(new DenseNode(in, channel_n, add_bias, group_n));
  return nodes_.size() - 1;
}

Builder::NodeId Builder::BatchNormalization(Builder::NodeId in) {
  CheckId(in);
  nodes_.emplace_back(new BatchNormalizationNode(in));
  return nodes_.size() - 1;
}

Builder::NodeId Builder::Swish(Builder::NodeId in) {
  CheckId(in);
  nodes_.emplace_back(new SwishNode(in));
  return nodes_.size() - 1;
}

Builder::NodeId Builder::Tanh(Builder::NodeId in) {
  CheckId(in);
  nodes_.emplace_back(new TanhNode(in));
  return nodes_.size() - 1;
}

Builder::NodeId Builder::Sigmoid(Builder::NodeId in) {
  CheckId(in);
  nodes_.emplace_back(new SigmoidNode(in));
  return nodes_.size() - 1;
}

Builder::NodeId Builder::Softmax(Builder::NodeId in) {
  CheckId(in);
  nodes_.emplace_back(new SoftmaxNode(in));
  return nodes_.size() - 1;
}

Builder::NodeId Builder::Add(Builder::NodeId in_a, Builder::NodeId in_b) {
  CheckId(in_a);
  CheckId(in_b);
  nodes_.emplace_back(new AddNode(in_a, in_b));
  return nodes_.size() - 1;
}

void Builder::Reset() {
  inputs_.clear();
  nodes_.clear();
}

}  // namespace nn
}  // namespace azah
