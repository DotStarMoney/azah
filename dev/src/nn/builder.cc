#include "nn/builder.h"

#include <memory>
#include <string_view>

#include "glog/logging.h"

namespace azah {
namespace nn {

using std::string_view;

void Builder::CheckNotOutput(Builder::Type type) {
  // glog doesn't understand enum comparison
  CHECK_NE(static_cast<int>(type), static_cast<int>(Builder::Type::kOutput));
}

void Builder::CheckId(NodeId id) const {
  CHECK_LT(id, nodes_.size());
  CheckNotOutput(nodes_[id]->type);
}

Builder::NodeId Builder::InsertNode(std::unique_ptr<BuilderNode> node,
                                    NodeId in) {
  CheckId(in);
  nodes_.emplace_back(std::move(node));
  auto id = nodes_.size() - 1;
  nodes_[in]->outputs.push_back(id);
  return id;
}

Builder::NodeId Builder::InsertNode(std::unique_ptr<BuilderNode> node, 
                                    NodeId in_a, NodeId in_b) {
  CheckId(in_a);
  CheckId(in_b);
  nodes_.emplace_back(std::move(node));
  auto id = nodes_.size() - 1;
  nodes_[in_a]->outputs.push_back(id);
  nodes_[in_b]->outputs.push_back(id);
  return id;
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
  auto id = nodes_.size() - 1;
  nodes_[in]->outputs.push_back(id);
  outputs_.push_back(id);
}

Builder::NodeId Builder::Dense(Builder::NodeId in, int channel_n, bool add_bias,
                               int group_n) {
  return InsertNode(
      std::make_unique<DenseNode>(in, channel_n, add_bias, group_n), in);
}

Builder::NodeId Builder::BatchNormalization(Builder::NodeId in) {
  return InsertNode(std::make_unique<BatchNormalizationNode>(in), in);
}

Builder::NodeId Builder::Swish(Builder::NodeId in) {
  return InsertNode(std::make_unique<SwishNode>(in), in);
}

Builder::NodeId Builder::Tanh(Builder::NodeId in) {
  return InsertNode(std::make_unique<TanhNode>(in), in);
}

Builder::NodeId Builder::Sigmoid(Builder::NodeId in) {
  return InsertNode(std::make_unique<SigmoidNode>(in), in);
}

Builder::NodeId Builder::Softmax(Builder::NodeId in) {
  return InsertNode(std::make_unique<SoftmaxNode>(in), in);
}

Builder::NodeId Builder::Add(Builder::NodeId in_a, Builder::NodeId in_b) {
  return InsertNode(std::make_unique<AddNode>(in_a, in_b), in_a, in_b);
}

Builder::NodeId Builder::Subtract(Builder::NodeId in_a, Builder::NodeId in_b) {
  return InsertNode(std::make_unique<SubtractNode>(in_a, in_b), in_a, in_b);
}

Builder::NodeId Builder::Multiply(Builder::NodeId in_a, Builder::NodeId in_b) {
  return InsertNode(std::make_unique<MultiplyNode>(in_a, in_b), in_a, in_b);
}

Builder::NodeId Builder::Multiply(Builder::NodeId in, float constant) {
  return InsertNode(std::make_unique<MultiplyNode>(in, constant), in);
}

Builder::NodeId Builder::Divide(Builder::NodeId in_a, Builder::NodeId in_b) {
  return InsertNode(std::make_unique<DivideNode>(in_a, in_b), in_a, in_b);
}

Builder::NodeId Builder::Divide(Builder::NodeId in, float constant) {
  return InsertNode(std::make_unique<DivideNode>(in, constant), in);
}

Builder::NodeId Builder::Power(Builder::NodeId in, float constant) {
  return InsertNode(std::make_unique<PowerNode>(in, constant), in);
}

void Builder::Reset() {
  inputs_.clear();
  outputs_.clear();
  nodes_.clear();
}

}  // namespace nn
}  // namespace azah
