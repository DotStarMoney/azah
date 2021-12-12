#ifndef AZAH_NN_GRAPH_BUILDER_H_
#define AZAH_NN_GRAPH_BUILDER_H_

#include <initializer_list>
#include <memory>
#include <string_view>
#include <tuple>

namespace azah {
namespace nn {
namespace graph {

struct BuilderNode;

// An input node.
std::unique_ptr<BuilderNode> Input(std::string_view input_name, int channel_n);

// An output node.
BuilderNode* Output(BuilderNode* in, std::string_view output_name);

BuilderNode* Dense(BuilderNode* in, int channel_n, bool add_bias = false, 
                   int group_n = 1);

//
//
//

std::unique_ptr<BuilderNode> BatchNormalization(std::unique_ptr<BuilderNode> in);
std::unique_ptr<BuilderNode> Swish(std::unique_ptr<BuilderNode> in);
std::unique_ptr<BuilderNode> Tanh(std::unique_ptr<BuilderNode> in);
std::unique_ptr<BuilderNode> Sigmoid(std::unique_ptr<BuilderNode> in);
std::unique_ptr<BuilderNode> Softmax(std::unique_ptr<BuilderNode> in);
std::unique_ptr<BuilderNode> Add(
    std::initializer_list<std::unique_ptr<BuilderNode>> in);

}  // namespace graph
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_GRAPH_BUILDER_H_
