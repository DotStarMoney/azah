#include "network.h"

#include <vector>

#include "data_types.h"

namespace azah {
namespace nn {

void Network::Outputs(const std::vector<uint32_t>& outputs_i,
                      std::vector<DynamicMatrixRef>& outputs) {

}

void Network::Gradients(const std::vector<uint32_t>& targets_i,
                        std::vector<DynamicMatrixRef>& variables,
                        std::vector<float>& losses) {

}

void Network::SetVariables(const std::vector<DynamicMatrixRef>& variables) {

}
  
void Network::GetVariables(std::vector<DynamicMatrixRef>& variables) {

}

void Network::SetConstants(const std::vector<uint32_t>& constants_i,
                           const std::vector<DynamicMatrixRef>& constants) {

}

}  // namespace nn
}  // namespace azah
