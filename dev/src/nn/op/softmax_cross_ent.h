#ifndef AZAH_NN_OPS_SOFTMAX_CROSS_ENT_H_
#define AZAH_NN_OPS_SOFTMAX_CROSS_ENT_H_

#include <stdint.h>

#include "../binary_op.h"
#include "../data_types.h"
#include "../node.h"
#include "softmax.h"

namespace azah {
namespace nn {
namespace op {

template <int Rows, int Cols>
class SoftmaxCrossEnt : public BinaryOp<Rows, Cols, Rows, Cols, 1, 1> {
 public:
  SoftmaxCrossEnt(const SoftmaxCrossEnt&) = delete;
  SoftmaxCrossEnt& operator=(const SoftmaxCrossEnt&) = delete;

  SoftmaxCrossEnt(Node<Rows, Cols>& input_a, Node<Rows, Cols>& input_b)
      : BinaryOp<Rows, Cols, Rows, Cols, 1, 1>(input_a, input_b) {}

  void Backprop(uint32_t cycle, const MatrixRef<1, 1>& output_dx) override {
    auto c = output_dx.value();
    auto pred_softmax = Softmax<Rows, Cols>::SoftmaxExpr(
        this->input_a_.Output(cycle)).eval();
    if (!this->input_a_.constant) {
      const auto& troo = this->input_b_.Output(cycle);
      this->input_a_.Backprop(
          cycle, 
          (c * (pred_softmax - troo.array())).matrix());
    }
    if (!this->input_b_.constant) {
      this->input_b_.Backprop(cycle, (c * -pred_softmax.log()).matrix());
    }
  }

 private:
  void ComputeOutput(uint32_t cycle) override {
    auto pred_softmax = Softmax<Rows, Cols>::SoftmaxExpr(
        this->input_a_.Output(cycle));
    const auto& troo = this->input_b_.Output(cycle);
    auto log_like = troo.array() * pred_softmax.log();
    this->cached_output_ = Matrix<1, 1>::Constant(-log_like.sum());
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_OPS_SOFTMAX_CROSS_ENT_H_
