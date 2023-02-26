#ifndef AZAH_NN_OPS_CONCAT_H_
#define AZAH_NN_OPS_CONCAT_H_

#include <stdint.h>

#include "../binary_op.h"
#include "../data_types.h"
#include "../node.h"

namespace azah {
namespace nn {
namespace op {

template <int InputRowsA, int InputRowsB, int InputCols>
class Concat : public BinaryOp<InputRowsA, InputCols, InputRowsB, InputCols, 
                               InputRowsA + InputRowsB, InputCols> {
 public:
  Concat(const Concat&) = delete;
  Concat& operator=(const Concat&) = delete;

  Concat(Node<InputRowsA, InputCols>& input_a, 
         Node<InputRowsB, InputCols>& input_b)
      : BinaryOp<InputRowsA, InputCols, InputRowsB, InputCols, 
                 InputRowsA + InputRowsB, InputCols>(input_a, input_b) {}

  void Backprop(
      uint32_t cycle,
      const MatrixRef<InputRowsA + InputRowsB, 
                      InputCols>& output_dx) override {
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
