#ifndef AZAH_NN_VARIABLE_H_
#define AZAH_NN_VARIABLE_H_

#include <stdint.h>

#include "data_types.h"
#include "node.h"
#include "variable_base.h"

namespace azah {
namespace nn {

template <int Rows, int Cols>
class Variable : public Node<Rows, Cols>, public VariableBase {
 public:
  Variable(const Variable&) = delete;
  Variable& operator=(const Variable&) = delete;

  Variable(const MatrixRef<Rows, Cols>& x) 
      : Node<Rows, Cols>(false), 
        value_(x),
        gradient_(Matrix<Rows, Cols>::Zero()), 
        grad_cycle_(-1) {}

  ConstDynamicMatrixRef gradient_base() const override {
    return gradient_;
  }

  DynamicMatrixRef value_base() override {
    return value_;
  }

  bool updated(uint32_t cycle) const {
    return grad_cycle_ == cycle;
  }

  const Matrix<Rows, Cols>& Output(uint32_t cycle) override {
    return value_;
  }

  void Backprop(uint32_t cycle, const MatrixRef<Rows, Cols>& output_dx) override {
    if (cycle != grad_cycle_) {
      gradient_ = output_dx;
      grad_cycle_ = cycle;
    } else {
      gradient_ += output_dx;
    }
  }

 private:
  Matrix<Rows, Cols> gradient_;
  uint32_t grad_cycle_;
  Matrix<Rows, Cols> value_;

};

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_VARIABLE_H_
