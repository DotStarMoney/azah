#ifndef AZAH_NN_OPS_SCALAR_MSE_H_
#define AZAH_NN_OPS_SCALAR_MSE_H_

#include <stdint.h>

#include "../binary_op.h"
#include "../data_types.h"
#include "../node.h"

namespace azah {
namespace nn {
namespace op {

template <int Rows, int Cols>
class ScalarMSE : public BinaryOp<Rows, Cols, 1, 1, 1, 1> {
 public:
  ScalarMSE(const ScalarMSE&) = delete;
  ScalarMSE& operator=(const ScalarMSE&) = delete;

  ScalarMSE(Node<Rows, Cols>& input, Node<1, 1>& target) :
      BinaryOp<Rows, Cols, 1, 1, 1, 1>(input, target) {}

  void Backprop(
      uint32_t cycle,
      const MatrixRef<1, 1>& output_dx = Matrix<1, 1>::Constant(1)) override {
    auto x = this->input_a_.Output(cycle);
    auto target = this->input_b_.Output(cycle);
    auto lhs_prod_array = output_dx.value() * (x.array() - target.value())
        * static_cast<float>(2.0 / (Rows * Cols));
    if (!this->input_a_.constant) {
      this->input_a_.Backprop(cycle, lhs_prod_array.matrix());
    }
    if (!this->input_b_.constant) {
      this->input_b_.Backprop(cycle,
                              Matrix<1, 1>::Constant(-lhs_prod_array.sum()));
    }
  }

 private:
  void ComputeOutput(uint32_t cycle) override {
    auto x = this->input_a_.Output(cycle);
    auto target = this->input_b_.Output(cycle);
    this->cached_output_ =
        Matrix<1, 1>::Constant((x.array() - target.value()).square().mean());
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_OPS_SCALAR_MSE_H_
