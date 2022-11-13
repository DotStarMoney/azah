#ifndef AZAH_NN_CONSTANT_BASE_H_
#define AZAH_NN_CONSTANT_BASE_H_

#include <stdint.h>

#include "data_types.h"

namespace azah {
namespace nn {

class ConstantBase {
 public:
  ConstantBase(const ConstantBase&) = delete;
  ConstantBase& operator=(const ConstantBase&) = delete;

  virtual DynamicMatrixRef value_base() = 0;

 protected:
  ConstantBase() {}
};

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_CONSTANT_BASE_H_
