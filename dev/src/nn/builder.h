#ifndef AZAH_NN_BUILDER_H_
#define AZAH_NN_BUILDER_H_

#include <stddef.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/strings/substitute.h"

namespace azah {
namespace nn {

class Builder {
 public:
  typedef std::size_t NodeId;

  // An input node.
  NodeId Input(std::string_view input_name, int channel_n);

  // An output node.
  void Output(NodeId in, std::string_view output_name);
  
  // A fully connected layer, optionally grouping to create sparsity.
  // E.g. if group = 2, each output is a function of channels_n / group inputs.
  NodeId Dense(NodeId in, int channel_n, bool add_bias = false, 
               int group_n = 1);

  // Batch normalization.
  NodeId BatchNormalization(NodeId in);

  // Activations.
  NodeId Swish(NodeId in);
  NodeId Tanh(NodeId in);
  NodeId Sigmoid(NodeId in);
  NodeId Softmax(NodeId in);

  // Arithmetic functions.
  NodeId Add(NodeId in_a, NodeId in_b);

  void Reset();

  Builder(const Builder&) = delete;
  Builder& operator=(const Builder&) = delete;

 private:
  enum class Type {
     kInput,
     kOutput,
     kDense,
     kBatchNormalization,
     kSwish,
     kTanh,
     kSigmoid,
     kSoftmax,
     kAdd
  };

  struct BuilderNode {
    BuilderNode(std::string&& description, Type type,
        std::initializer_list<Builder::NodeId> inputs_) :
        description(description), type(type), inputs(inputs) {}

    BuilderNode(const BuilderNode&) = delete;
    BuilderNode& operator=(const BuilderNode&) = delete;

    const std::string description;
    const Type type;
    const std::vector<Builder::NodeId> inputs;

    std::vector<Builder::NodeId> outputs;
  };

  struct InputNode : public BuilderNode {
    InputNode(std::string_view input_name, int channel_n) :
        BuilderNode(Description(input_name, channel_n), Type::kInput, {}),
                    input_name(input_name), channel_n(channel_n) {}

    std::string Description(std::string_view input_name, int channels_n) const {
      return absl::Substitute("Input<name={}, channels_n={}, ptr={}>", 
                              input_name, channels_n, this);
    }

    const std::string input_name;
    const int channel_n;
  };

  struct OutputNode : public BuilderNode {
    OutputNode(NodeId in, std::string_view output_name) :
        BuilderNode(Description(output_name), Type::kOutput, {in}),
        output_name(output_name) {}

    std::string Description(std::string_view output_name) const {
      return absl::Substitute("Output<name={}, ptr={}>", output_name, this);
    }

    const std::string output_name;
  };

  struct DenseNode : public BuilderNode {
    DenseNode(NodeId in, int channel_n, bool add_bias, int group_n) :
        BuilderNode(Description(channel_n, add_bias, group_n), Type::kDense, 
                    {in}),
        channel_n(channel_n),
        add_bias(add_bias),
        group_n(group_n) {}

    std::string Description(int channels_n, bool add_bias, int group_n) const {
      return absl::Substitute(
          "Dense<channels_n={}, add_bias={}, group_n={}, ptr={}>", channels_n,
          add_bias, group_n, this);
    }

    const int channel_n;
    const bool add_bias;
    const bool group_n;
  };

  struct BatchNormalizationNode : public BuilderNode {
    BatchNormalizationNode(NodeId in) :
        BuilderNode(Description(), Type::kBatchNormalization, {in}) {}

    std::string Description() const {
      return absl::Substitute("BatchNormalization<ptr={}>", this);
    }
  };

  struct SwishNode : public BuilderNode {
    SwishNode(NodeId in) : BuilderNode(Description(), Type::kSwish, {in}) {}

    std::string Description() const {
      return absl::Substitute("Swish<ptr={}>", this);
    }
  };

  struct TanhNode : public BuilderNode {
    TanhNode(NodeId in) : BuilderNode(Description(), Type::kTanh, {in}) {}

    std::string Description() const {
      return absl::Substitute("Tanh<ptr={}>", this);
    }
  };

  struct SigmoidNode : public BuilderNode {
    SigmoidNode(NodeId in) : BuilderNode(Description(),Type::kSigmoid, {in}) {}

    std::string Description() const {
      return absl::Substitute("Sigmoid<ptr={}>", this);
    }
  };

  struct SoftmaxNode : public BuilderNode {
    SoftmaxNode(NodeId in) : BuilderNode(Description(), Type::kSoftmax, {in}) {}

    std::string Description() const {
      return absl::Substitute("Softmax<ptr={}>", this);
    }
  };

  struct AddNode : public BuilderNode {
    AddNode(NodeId in_a, NodeId in_b) : 
        BuilderNode(Description(), Type::kAdd, {in_a, in_b}) {}

    std::string Description() const {
      return absl::Substitute("Add<ptr={}>", this);
    }
  };

  void CheckId(NodeId id) const;

  std::vector<NodeId> inputs_;
  std::vector<std::unique_ptr<BuilderNode>> nodes_;
};

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_BUILDER_H_
