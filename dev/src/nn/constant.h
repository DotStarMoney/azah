#ifndef AZAH_NN_CONSTANT_H_
#define AZAH_NN_CONSTANT_H_

#include <stdint.h>

#include "constant_base.h"
#include "data_types.h"
#include "glog/logging.h"
#include "node.h"

namespace azah {
namespace nn {

template <int Rows, int Cols>
class Constant : public Node<Rows, Cols>, public ConstantBase {
 public:
  Constant(const Constant&) = delete;
  Constant& operator=(const Constant&) = delete;

  Constant(const MatrixRef<Rows, Cols>& x) : Node<Rows, Cols>(true), 
                                             constant_value_(x) {}

  DynamicMatrixRef value_base() {
    return constant_value_;
  }

  const Matrix<Rows, Cols>& Output(uint32_t cycle) override {
    return constant_value_;
  }

  void Backprop(uint32_t cycle, const MatrixRef<Rows, Cols>& output_dx) override {
    LOG(FATAL) << "Cannot propagate gradients to a constant.";
  }

 protected:
  Matrix<Rows, Cols> constant_value_;
};

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_CONSTANT_H_
