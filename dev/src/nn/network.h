#ifndef AZAH_NN_NETWORK_H_
#define AZAH_NN_NETWORK_H_

#include <stdint.h>

#include <vector>

#include "constant_base.h"
#include "data_types.h"
#include "node_base.h"
#include "variable_base.h"

namespace azah {
namespace nn {

class Network {
 public:
  Network(const Network&) = delete;
  Network& operator=(const Network&) = delete;

  void Outputs(const std::vector<uint32_t>& outputs_i, 
               std::vector<DynamicMatrix>& outputs);

  void Gradients(const std::vector<uint32_t>& targets_i,
                 std::vector<uint32_t>& variables_i,
                 std::vector<DynamicMatrix>& gradients,
                 std::vector<float>& losses);

  void SetVariables(const std::vector<uint32_t>& variables_i, 
                    const std::vector<DynamicMatrix>& variables);

  // Leave variables_i empty to retrieve all variables.
  void GetVariables(const std::vector<uint32_t>& variables_i, 
                    std::vector<DynamicMatrixRef>& variables);

  // Leave variables_i empty to retrieve all variables.
  void GetVariables(const std::vector<uint32_t>& variables_i,
                    std::vector<ConstDynamicMatrixRef>& variables) const;
  
  void SetConstants(const std::vector<uint32_t>& constants_i, 
                    const std::vector<DynamicMatrix>& constants);

  void GetConstantsByTag(int tag,
                         std::vector<int>& constants_i,
                         std::vector<DynamicMatrixRef>& constants);

  void GetConstantsByTag(int tag, std::vector<DynamicMatrixRef>& constants);

 protected:
  Network();

  void AddOutput(NodeBase* output);
  void AddTarget(NodeBase* target);
  void AddVariable(VariableBase* variable);
  void AddConstant(ConstantBase* constant, int tag);

 private:
  uint32_t cycle_;
  
  std::vector<NodeBase*> outputs_;
  
  std::vector<NodeBase*> targets_;
  
  std::vector<VariableBase*> variables_;

  std::vector<ConstantBase*> constants_;
  std::vector<int> constant_tags_;
};

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_NETWORK_H_
