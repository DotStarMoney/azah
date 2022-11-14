#include "network.h"

#include <vector>

#include "data_types.h"
#include "glog/logging.h"

namespace azah {
namespace nn {

void Network::Outputs(const std::vector<uint32_t>& outputs_i,
                      std::vector<ConstDynamicMatrixRef>& outputs) {
  if (outputs_i.empty()) {
    LOG(FATAL) << "\"outputs_i\" cannot be empty.";
  }
  outputs.clear();
  for (auto output_i : outputs_i) {
    outputs.push_back(outputs_[output_i]->OutputBase(cycle_));
  }
  ++cycle_;
}

void Network::Gradients(const std::vector<uint32_t>& targets_i,
                        std::vector<uint32_t>& variables_i,
                        std::vector<ConstDynamicMatrixRef>& gradients,
                        std::vector<float>& losses) {
  if (targets_i.empty()) {
    LOG(FATAL) << "\"targets_i\" cannot be empty.";
  }

  losses.clear();
  for (auto target_i : targets_i) {
    auto target = targets_[target_i];
    target->BackpropBase(cycle_);
    losses.push_back(target->OutputBase(cycle_).value());
  }

  variables_i.clear();
  gradients.clear();
  for (int i = 0; i < variables_.size(); ++i) {
    auto var = variables_[i];
    if (!var->updated(cycle_)) continue;
    variables_i.push_back(i);
    gradients.push_back(var->gradient_base());
  }

  ++cycle_;
}

void Network::SetVariables(const std::vector<uint32_t>& variables_i,
                           const std::vector<DynamicMatrixRef>& variables) {
  if (variables_i.empty()) {
    LOG(FATAL) << "\"variables_i\" cannot be empty.";
  }
  for (int i = 0; i < variables_i.size(); ++i) {
    auto variable_i = variables_i[i];
    constants_[variable_i]->value_base() = variables[i];
  }
}

void Network::GetVariables(const std::vector<uint32_t>& variables_i,
                           std::vector<DynamicMatrixRef>& variables) {
  variables.clear();
  if (variables_i.empty()) {
    for (auto var : variables_) {
      variables.push_back(var->value_base());
    }
  } else {
    for (auto var_i : variables_i) {
      variables.push_back(variables_[var_i]->value_base());
    }
  }
}

void Network::GetVariables(const std::vector<uint32_t>& variables_i,
                           std::vector<ConstDynamicMatrixRef>& variables) const {
  variables.clear();
  if (variables_i.empty()) {
    for (auto var : variables_) {
      variables.push_back(var->value_base());
    }
  }
  else {
    for (auto var_i : variables_i) {
      variables.push_back(variables_[var_i]->value_base());
    }
  }
}

void Network::SetConstants(const std::vector<uint32_t>& constants_i,
                           const std::vector<DynamicMatrixRef>& constants) {
  if (constants_i.empty()) {
    LOG(FATAL) << "\"constants_i\" cannot be empty.";
  }
  for (int i = 0; i < constants_i.size(); ++i) {
    auto constant_i = constants_i[i];
    constants_[constant_i]->value_base() = constants[i];
  }
}

Network::Network() : cycle_(0) {}

void Network::AddOutput(NodeBase* output) {
  outputs_.push_back(output);
}

void Network::AddTarget(NodeBase* target) {
  if (target->size() != 1) {
    LOG(FATAL) << "Cannot add gradient target with more than one element.";
  }
  targets_.push_back(target);
}

void Network::AddVariable(VariableBase* variable) {
  variables_.push_back(variable);
}

void Network::AddConstant(ConstantBase* constant) {
  constants_.push_back(constant);
}

}  // namespace nn
}  // namespace azah
