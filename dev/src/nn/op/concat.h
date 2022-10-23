#ifndef AZAH_NN_OPS_CONCAT_H_
#define AZAH_NN_OPS_CONCAT_H_

#include <stdint.h>

#include "../binary_op.h"
#include "../data_types.h"
#include "../node.h"

namespace azah {
namespace nn {
namespace op {

template <int InputRowsA, int InputColsA, int InputRowsB, int InputColsB>
class Concat : public BinaryOp<InputRowsA, InputColsA, InputRowsB, InputColsB, 
                               InputRowsA + InputRowsB, InputColsA> {
  static_assert(InputColsA == InputColsB, 
                "Inputs must have the same number of columns.");
 public:
  Concat(const Concat&) = delete;
  Concat& operator=(const Concat&) = delete;

  Concat(Node<InputRowsA, InputColsA>& input_a, 
         Node<InputRowsB, InputColsB>& input_b)
      : BinaryOp<InputRowsA, InputColsA, InputRowsB, InputColsB, 
                 InputRowsA + InputRowsB, InputColsA>(input_a, input_b) {}

  void Backprop(
      uint32_t cycle,
      const MatrixRef<InputRowsA + InputRowsB, InputColsA>& output_dx =
          Matrix<InputRowsA + InputRowsB, InputColsA>::Constant(1)) override {
    if (!this->input_a_.constant) {
      this->input_a_.Backprop(cycle, output_dx.topRows(InputRowsA));
    }
    if (!this->input_b_.constant) {
      this->input_b_.Backprop(cycle, output_dx.bottomRows(InputRowsB));
    }
  }

 private:
  void ComputeOutput(uint32_t cycle) override {
    this->cached_output_.topRows(InputRowsA) = this->input_a_.Output(cycle);
    this->cached_output_.bottomRows(InputRowsB) = this->input_b_.Output(cycle);
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_OPS_CONCAT_H_
