#ifndef AZAH_NN_NETWORK_H_
#define AZAH_NN_NETWORK_H_

#include <stdint.h>

#include <vector>

#include "data_types.h"
#include "node.h"

namespace azah {
namespace nn {

class Network {
 public:
  Network(const Network&) = delete;
  Network& operator=(const Network&) = delete;

  void Outputs(const std::vector<uint32_t>& outputs_i, 
               std::vector<DynamicMatrixRef>& outputs);

  void Gradients(const std::vector<uint32_t>& targets_i,
                 std::vector<DynamicMatrixRef>& variables,
                 std::vector<float>& losses);

  void SetVariables(const std::vector<DynamicMatrixRef>& variables);
  void GetVariables(std::vector<DynamicMatrixRef>& variables);

  void SetConstants(const std::vector<uint32_t>& constants_i, 
                    const std::vector<DynamicMatrixRef>& constants);

 protected:
  uint32_t cycle_;
 
  Network(uint32_t outputs_n);
};

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_NETWORK_H_
