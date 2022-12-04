#ifndef AZAH_NN_OPS_MSE_H_
#define AZAH_NN_OPS_MSE_H_

#include <stdint.h>

#include "../binary_op.h"
#include "../data_types.h"
#include "../node.h"

namespace azah {
namespace nn {
namespace op {

template <int Rows, int Cols>
class MSE : public BinaryOp<Rows, Cols, Rows, Cols, 1, 1> {
public:
  MSE(const MSE&) = delete;
  MSE& operator=(const MSE&) = delete;

  MSE(Node<Rows, Cols>& input_a, Node<Rows, Cols>& input_b)
      : BinaryOp<Rows, Cols, Rows, Cols, 1, 1>(input_a, input_b) {}

  void Backprop(uint32_t cycle, const MatrixRef<1, 1>& output_dx) override {
    auto a = this->input_a_.Output(cycle);
    auto b = this->input_b_.Output(cycle);
    auto dmse = 
        output_dx.value() * 2.0f / static_cast<float>(Rows * Cols) * (a - b);
    if (!this->input_a_.constant) {
      this->input_a_.Backprop(cycle, dmse);
    }
    if (!this->input_b_.constant) {
      this->input_b_.Backprop(cycle, -dmse);
    }
  }

protected:
  void ComputeOutput(uint32_t cycle) override {
    auto a = this->input_a_.Output(cycle);
    auto b = this->input_b_.Output(cycle);
    this->cached_output_ = 
        Matrix<1, 1>::Constant((a.array() - b.array()).square().mean());
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_OPS_MSE_H_
