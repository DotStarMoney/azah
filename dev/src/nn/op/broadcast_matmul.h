#ifndef AZAH_NN_OPS_BROADCAST_MATMUL_H_
#define AZAH_NN_OPS_BROADCAST_MATMUL_H_

#include <stdint.h>

#include "../binary_op.h"
#include "../data_types.h"
#include "../init.h"
#include "../node.h"

namespace azah {
namespace nn {
namespace op {

template <int InputRowsA, int InputColsA, int InputRowsB, int OutputCols>
class BroadcastMatmul : public BinaryOp<InputRowsA, InputColsA, InputRowsB, 1,
                                        InputRowsA / OutputCols, OutputCols> {
  static_assert(InputRowsA % InputRowsOut == 0,
                "Output columns must divide the rows of A.");
 public:
  BroadcastMatmul(const BroadcastMatmul&) = delete;
  BroadcastMatmul& operator=(const BroadcastMatmul&) = delete;

  BroadcastMatmul(Node<InputRowsA, InputColsA>& input_a,
                  Node<InputRowsB, 1>& input_b) :
      BinaryOp<InputRowsA, InputColsA, InputRowsB, 1, InputRowsA / OutputCols,
               OutputColsB>(input_a, input_b) {}

  void Backprop(
      uint32_t cycle,
      const MatrixRef<InputRowsA / OutputCols, OutputCols>& output_dx) override {
    if (!this->input_a_.constant) {
      Matrix<InputRowsA, InputColsA> j;
      auto b_trans = this->input_b_.Output(cycle).transpose();
      for (int c = 0; c < OutputCols; ++c) {
        j.middleRows(c * (InputRowsA / OutputCols), InputRowsA / OutputCols) = 
            output_dx.col(c) * b_trans;
      }
      this->input_a_.Backprop(cycle, j);
    }
    if (!this->input_b_.constant) {
      Matrix<InputRowsB, 1> j = init::Zeros<InputRowsB, 1>();
      auto a_trans = this->input_a_.Output(cycle).transpose();
      for (int c = 0; c < OutputCols; ++c) {
        j += a_trans.middleCols(
            c * (InputRowsA / OutputCols), 
            InputRowsA / OutputCols) * output_dx.col(c);
      }
      this->input_b_.Backprop(cycle, j);
    }
  }

 private:
  void ComputeOutput(uint32_t cycle) override {
    auto a = this->input_a_.Output(cycle);
    auto b = this->input_b_.Output(cycle);
    for (int c = 0; c < OutputCols; ++c) {
      this->cached_output_.col(c) = 
          a.middleRows(c * (InputRowsA / OutputCols), InputRowsA / OutputCols) * b;
    }
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_OPS_BROADCAST_MATMUL_H_
