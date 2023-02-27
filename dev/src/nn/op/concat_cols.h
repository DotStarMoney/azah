#ifndef AZAH_NN_OPS_CONCAT_COLS_H_
#define AZAH_NN_OPS_CONCAT_COLS_H_

#include <stdint.h>

#include "../binary_op.h"
#include "../data_types.h"
#include "../node.h"

namespace azah {
namespace nn {
namespace op {

template <int InputRows, int InputColsA, int InputColsB>
class ConcatCols : public BinaryOp<InputRows, InputColsA, InputRows, InputColsB, 
                                   InputRows, InputColsA + InputColsB> {
 public:
  ConcatCols(const ConcatCols&) = delete;
  ConcatCols& operator=(const ConcatCols&) = delete;

  ConcatCols(Node<InputRows, InputColsA>& input_a,
             Node<InputRows, InputColsB>& input_b)
      : BinaryOp<InputRows, InputColsA, InputRows, InputColsB, 
                 InputRows, InputColsA + InputColsB>(input_a, input_b) {}

  void Backprop(
      uint32_t cycle,
      const MatrixRef<InputRows, InputColsA + InputColsB>& output_dx) override {
    if (!this->input_a_.constant) {
      this->input_a_.Backprop(cycle, output_dx.leftCols(InputColsA));
    }
    if (!this->input_b_.constant) {
      this->input_b_.Backprop(cycle, output_dx.rightCols(InputColsB));
    }
  }

 private:
  void ComputeOutput(uint32_t cycle) override {
    this->cached_output_.leftCols(InputColsA) = this->input_a_.Output(cycle);
    this->cached_output_.rightCols(InputColsB) = this->input_b_.Output(cycle);
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_OPS_CONCAT_COLS_H_
