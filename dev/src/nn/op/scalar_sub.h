#ifndef AZAH_NN_OPS_SCALAR_SUB_H_
#define AZAH_NN_OPS_SCALAR_SUB_H_

#include <stdint.h>

#include "../binary_op.h"
#include "../data_types.h"
#include "../node.h"

namespace azah {
namespace nn {
namespace op {

template <int Rows, int Cols>
class ScalarSub : public BinaryOp<Rows, Cols, 1, 1, Rows, Cols> {
 public:
  ScalarSub(const ScalarSub&) = delete;
  ScalarSub& operator=(const ScalarSub&) = delete;

  ScalarSub(Node<Rows, Cols>& input_a, Node<1, 1>& input_b)
      : BinaryOp<Rows, Cols, 1, 1, Rows, Cols>(input_a, input_b) {}

  void Backprop(uint32_t cycle, const MatrixRef<Rows, Cols>& output_dx) override {
    if (!this->input_a_.constant) {
      this->input_a_.Backprop(cycle, output_dx);
    }
    if (!this->input_b_.constant) {
      this->input_b_.Backprop(cycle, Matrix<1, 1>::Constant(-output_dx.sum()));
    }
  }

 protected:
  void ComputeOutput(uint32_t cycle) override {
    const auto& a = this->input_a_.Output(cycle);
    const auto& b = this->input_b_.Output(cycle);
    this->cached_output_ = (a.array() - b.value()).matrix();
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_OPS_SCALAR_SUB_H_
