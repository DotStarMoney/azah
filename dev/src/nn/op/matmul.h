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
         Node<InputRowsB, InputColsB>& input_b) :
      BinaryOp<InputRowsA, InputColsA, InputRowsB, InputColsB, InputRowsA, 
               InputColsB>(input_a, input_b) {}

  void backprop(
      uint32_t cycle, 
      const MatrixRef<InputRowsA, InputColsB>& output_dx = Matrix<
          InputRowsA, InputColsB>::Constant(1)) {
    if (!this->input_a_.constant) {
      this->input_a_.backprop(
          cycle, 
          output_dx * this->input_b_.output(cycle).transpose());
    }
    if (!this->input_b_.constant) {
      this->input_b_.backprop(
          cycle,
          this->input_a_.output(cycle).transpose() * output_dx);
    }
  }

 protected:
  void compute_output(uint32_t cycle) {
    auto a = this->input_a_.output(cycle);
    auto b = this->input_b_.output(cycle);
    this->cached_output_ = a * b;
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_OPS_MATMUL_H_
