#ifndef AZAH_NN_OPS_SCALAR_INV_SQRT_H_
#define AZAH_NN_OPS_SCALAR_INV_SQRT_H_

#include <math.h>
#include <stdint.h>

#include "../data_types.h"
#include "../node.h"
#include "../binary_op.h"

namespace azah {
namespace nn {
namespace op {

template <int Rows, int Cols>
class ScalarInvSqrt : public BinaryOp<Rows, Cols, 1, 1, Rows, Cols> {
 public:
  ScalarInvSqrt(const ScalarInvSqrt&) = delete;
  ScalarInvSqrt& operator=(const ScalarInvSqrt&) = delete;

  ScalarInvSqrt(Node<Rows, Cols>& input_a, Node<1, 1>& input_b)
      : BinaryOp<Rows, Cols, 1, 1, Rows, Cols>(input_a, input_b) {}

  void Backprop(uint32_t cycle, const MatrixRef<Rows, Cols>& output_dx) override {
    auto recip_inv = 
        (this->input_b_.Output(cycle).array() + kEpsilon).inverse().value();
    auto recip_inv_sqrt = std::sqrt(recip_inv);

    if (!this->input_a_.constant) {
      this->input_a_.Backprop(cycle, 
                              (output_dx.array() * recip_inv_sqrt).matrix());
    }
    if (!this->input_b_.constant) {
      auto num = this->input_.Output(cycle).cwiseProduct(output_dx).array();
      this->input_b_.Backprop(cycle, Matrix<1, 1>::Constant(
          -num.sum().value() * 0.5 * recip_inv * recip_inv_sqrt));
    }
  }

 private:
  void ComputeOutput(uint32_t cycle) override {
    auto x = this->input_a_.Output(cycle);
    auto recip = this->input_b_.Output(cycle);
    this->cached_output_ = x.array() / (recip.array() + kEpsilon).sqrt().value();
  }

  static constexpr float kEpsilon = 1e-5;
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_OPS_SCALAR_INV_SQRT_H_
