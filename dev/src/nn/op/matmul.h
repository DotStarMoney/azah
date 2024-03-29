#ifndef AZAH_NN_OPS_MATMUL_H_
#define AZAH_NN_OPS_MATMUL_H_

#include <stdint.h>

#include "../binary_op.h"
#include "../data_types.h"
#include "../node.h"

namespace azah {
namespace nn {
namespace op {

template <int InputRowsA, int InputColsA, int InputRowsB, int InputColsB, 
          bool TransposeRHS = false>
class Matmul : public BinaryOp<InputRowsA, InputColsA, InputRowsB, InputColsB, 
                               InputRowsA, 
                               TransposeRHS ? InputRowsB : InputColsB> {
 public:
  Matmul(const Matmul&) = delete;
  Matmul& operator=(const Matmul&) = delete;

  Matmul(Node<InputRowsA, InputColsA>& input_a, 
         Node<InputRowsB, InputColsB>& input_b) :
      BinaryOp<InputRowsA, InputColsA, InputRowsB, InputColsB, InputRowsA, 
               TransposeRHS ? InputRowsB : InputColsB>(input_a, input_b) {}

  void Backprop(
      uint32_t cycle, 
      const MatrixRef<
          InputRowsA, 
          TransposeRHS ? InputRowsB : InputColsB>& output_dx) override {
    if (!this->input_a_.constant) {
      const auto& b = this->input_b_.Output(cycle);
      if constexpr (TransposeRHS) {
        this->input_a_.Backprop(cycle, output_dx * b);
      } else {
        this->input_a_.Backprop(cycle, output_dx * b.transpose());
      }
    }
    if (!this->input_b_.constant) {
      const auto& a = this->input_a_.Output(cycle);
      if constexpr (TransposeRHS) {
        this->input_b_.Backprop(
            cycle,
            (this->input_a_.Output(cycle).transpose() * output_dx).transpose());
      } else {
        this->input_b_.Backprop(
            cycle,
            this->input_a_.Output(cycle).transpose() * output_dx);
      }
    }
  }

 private:
  void ComputeOutput(uint32_t cycle) override {
    const auto& a = this->input_a_.Output(cycle);
    const auto& b = this->input_b_.Output(cycle);
    if constexpr (TransposeRHS) {
      this->cached_output_ = a * b.transpose();
    } else {
      this->cached_output_ = a * b;
    }
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_OPS_MATMUL_H_
