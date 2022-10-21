#ifndef AZAH_NN_VARIABLE_H_
#define AZAH_NN_VARIABLE_H_

#include <stdint.h>

#include "data_types.h"
#include "node.h"

namespace azah {
namespace nn {

template <int Rows, int Cols>
class Variable : public Node<Rows, Cols> {
 public:
  Variable(const Variable&) = delete;
  Variable& operator=(const Variable&) = delete;

  Variable(const MatrixRef<Rows, Cols>& x) 
      : Node<Rows, Cols>(false), 
        value(x), 
        gradient_(Matrix<Rows, Cols>::Zero()), 
        grad_cycle_(-1) {}

  const Matrix<Rows, Cols>& output(uint32_t cycle) override {
    return value;
  }

  void backprop(uint32_t cycle, const MatrixRef<Rows, Cols>& output_dx) override {
    if (cycle != grad_cycle_) {
      gradient_ = output_dx;
      grad_cycle_ = cycle;
    } else {
      gradient_ += output_dx;
    }
  }

  const Matrix<Rows, Cols> gradient() const {
    return gradient_;
  }

  Matrix<Rows, Cols> value;
 private:
  Matrix<Rows, Cols> gradient_;
  uint32_t grad_cycle_;
};

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_VARIABLE_H_
