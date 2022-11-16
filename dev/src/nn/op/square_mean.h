#ifndef AZAH_NN_OPS_SQUARE_MEAN_H_
#define AZAH_NN_OPS_SQUARE_MEAN_H_

#include <stdint.h>

#include "../activation.h"
#include "../data_types.h"
#include "../node.h"
#include "../unary_op.h"

namespace azah {
namespace nn {
namespace op {

template <int Rows, int Cols>
class SquareMean : public UnaryOp<Rows, Cols, 1, 1> {
 public:
  SquareMean(const SquareMean&) = delete;
  SquareMean& operator=(const SquareMean&) = delete;

  SquareMean(Node<Rows, Cols>& input) : UnaryOp<Rows, Cols, 1, 1>(input) {}
  
 private:
  void ComputeOutput(uint32_t cycle) override {
    auto x = this->input_.Output(cycle);
    this->cached_output_ = Matrix<1, 1>::Constant(x.array().square().mean());
  }

  void UnaryBackprop(uint32_t cycle, const MatrixRef<1, 1>& output_dx) override {
    auto x = this->input_.Output(cycle);
    this->input_.Backprop(
        cycle,
        x * ((2.0f * output_dx.value()) / static_cast<float>(Rows * Cols)));
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_OPS_SQUARE_MEAN_H_
