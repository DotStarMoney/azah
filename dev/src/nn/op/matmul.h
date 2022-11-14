#ifndef AZAH_NN_OPS_MATMUL_H_
#define AZAH_NN_OPS_MATMUL_H_

#include <stdint.h>

#include "../binary_op.h"
#include "../data_types.h"
#include "../node.h"

namespace azah {
namespace nn {
namespace op {

template <int InputRowsA, int InputColsA, int InputRowsB, int InputColsB>
class Matmul : public BinaryOp<InputRowsA, InputColsA, InputRowsB, InputColsB, 
                               InputRowsA, InputColsB> {
 public:
  Matmul(const Matmul&) = delete;
  Matmul& operator=(const Matmul&) = delete;

  Matmul(Node<InputRowsA, InputColsA>& input_a, 
         Node<InputRowsB, InputColsB>& input_b,
         bool transpose_rhs = false) :
      BinaryOp<InputRowsA, InputColsA, InputRowsB, InputColsB, InputRowsA, 
               InputColsB>(input_a, input_b),
      transpose_rhs_(transpose_rhs) {}

  void Backprop(
      uint32_t cycle, 
      const MatrixRef<InputRowsA, InputColsB>& output_dx = 
          Matrix<InputRowsA, InputColsB>::Constant(1)) override {
    if (!this->input_a_.constant) {
      auto b = this->input_b_.Output(cycle);
      this->input_a_.Backprop(
          cycle, 
          output_dx * (transpose_rhs_ ? b : b.transpose()));
    }
    if (!this->input_b_.constant) {
      this->input_b_.Backprop(
          cycle,
          this->input_a_.Output(cycle).transpose() * output_dx);
    }
  }

 private:
  const bool transpose_rhs_;

  void ComputeOutput(uint32_t cycle) override {
    auto a = this->input_a_.Output(cycle);
    auto b = this->input_b_.Output(cycle);
    this->cached_output_ = a * (transpose_rhs_ ? b.transpose() : b);
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_OPS_MATMUL_H_
