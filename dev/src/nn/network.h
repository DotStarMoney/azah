#ifndef AZAH_NN_NETWORK_H_
#define AZAH_NN_NETWORK_H_

#include <stdint.h>

#include <type_traits>
#include <vector>

#include "constant_base.h"
#include "data_types.h"
#include "glog/logging.h"
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

  // Leave variables_i empty to set all variables.
  template <typename SourceDynamicMatrix>
  void SetVariables(const std::vector<uint32_t>& variables_i,
                    const std::vector<SourceDynamicMatrix>& variables) {
    static_assert(
        std::is_same<SourceDynamicMatrix, DynamicMatrix>()
            || std::is_same<SourceDynamicMatrix, DynamicMatrixRef>()
            || std::is_same<SourceDynamicMatrix, ConstDynamicMatrixRef>(), 
        "Variable type must be a dynamic matrix.");
    if (variables_i.empty()) {
      if (variables.size() != variables_.size()) {
        LOG(FATAL) << "Number of provided variables does not match the number "
                      "of model variables.";
      }
      for (int i = 0; i < variables.size(); ++i) {
        variables_[i]->value_base() = variables[i];
      }
    }
    else {
      for (int i = 0; i < variables_i.size(); ++i) {
        variables_[variables_i[i]]->value_base() = variables[i];
      }
    }
  }

  // Leave variables_i empty to retrieve all variables.
  void GetVariables(const std::vector<uint32_t>& variables_i, 
                    std::vector<DynamicMatrixRef>& variables);

  // Leave variables_i empty to retrieve all variables.
  void GetVariables(const std::vector<uint32_t>& variables_i,
                    std::vector<ConstDynamicMatrixRef>& variables) const;
  
  void SetConstants(const std::vector<uint32_t>& constants_i, 
                    const std::vector<DynamicMatrix>& constants);

 protected:
  Network();

  void AddOutput(NodeBase* output);
  void AddTarget(NodeBase* target);
  void AddVariable(VariableBase* variable);
  void AddConstant(ConstantBase* constant);

 private:
  uint32_t cycle_;
  
  std::vector<NodeBase*> outputs_;
  
  std::vector<NodeBase*> targets_;
  
  std::vector<VariableBase*> variables_;

  std::vector<ConstantBase*> constants_;
};

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_NETWORK_H_
