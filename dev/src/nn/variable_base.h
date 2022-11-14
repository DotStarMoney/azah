#ifndef AZAH_NN_VARIABLE_BASE_H_
#define AZAH_NN_VARIABLE_BASE_H_

#include <stdint.h>

#include "data_types.h"

namespace azah {
namespace nn {

class VariableBase {
 public:
  VariableBase(const VariableBase&) = delete;
  VariableBase& operator=(const VariableBase&) = delete;

  virtual ConstDynamicMatrixRef gradient_base() const = 0;
  virtual DynamicMatrixRef value_base() = 0;
  virtual bool updated(uint32_t cycle) const = 0;

 protected:
  VariableBase() {}
};

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_VARIABLE_BASE_H_
