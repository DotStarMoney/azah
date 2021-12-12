#include "builder.h"

#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace azah {
namespace nn {
namespace graph {

using std::unique_ptr;

namespace {

enum class BuilderNodeType {
  kInput,
  kOutput,
  kDense,
  kBatchNormalization,
  kSwish,
  kTanh,
  kSigmoid,
  kSoftmax
};

}  // namespace

struct BuilderNode {
  BuilderNode(std::string&& description, BuilderNodeType type,
              std::initializer_list<const BuilderNode*> inputs_) :
      description(description), type(type), inputs(inputs) {}

  BuilderNode(const BuilderNode&) = delete;
  BuilderNode& operator=(const BuilderNode&) = delete;

  const std::string description;
  const BuilderNodeType type;
  const std::vector<const BuilderNode*> inputs;

  std::vector<const BuilderNode*> outputs;
};

namespace {

struct InputNode : public BuilderNode {
  InputNode(std::string_view input_name, int channels_n) :
      BuilderNode(Description(input_name, channels_n), BuilderNodeType::kInput,
                  {}),
      input_name(input_name),
      channels_n(channels_n) {}

  std::string Description(std::string_view input_name, int channels_n) {
    return "";
  }

  const std::string input_name;
  const int channels_n;
};


struct OutputNode : public BuilderNode {
  OutputNode(unique_ptr<BuilderNode> in, std::string_view output_name) :
      BuilderNode(Description(output_name), BuilderNodeType::kOutput,
      {std::move(in)}),
      input_name(input_name),
      channels_n(channels_n) {}

  std::string Description(std::string_view output_name) {
    return "";
  }

  const std::string input_name;
  const int channels_n;
};

}  // namespace

unique_ptr<BuilderNode> Input(std::string_view input_name, int channel_n) {
  return std::make_unique<InputNode>(input_name, channel_n);
}

unique_ptr<BuilderNode> Output(unique_ptr<BuilderNode> in, 
                               std::string_view output_name) {

}

unique_ptr<BuilderNode> Dense(unique_ptr<BuilderNode>, int channel_n, 
                              bool add_bias, int group_n) {

}

unique_ptr<BuilderNode> BatchNormalization(unique_ptr<BuilderNode> in) {

}

unique_ptr<BuilderNode> Swish(unique_ptr<BuilderNode> in) {

}

unique_ptr<BuilderNode> Tanh(unique_ptr<BuilderNode> in) {

}

unique_ptr<BuilderNode> Sigmoid(unique_ptr<BuilderNode> in) {

}

unique_ptr<BuilderNode> Softmax(unique_ptr<BuilderNode> in) {

}

unique_ptr<BuilderNode> Add(
    std::initializer_list<std::unique_ptr<BuilderNode>> in) {

}

}  // namespace graph
}  // namespace nn
}  // namespace azah
