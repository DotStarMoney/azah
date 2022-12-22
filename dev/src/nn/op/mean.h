#ifndef AZAH_NN_OPS_MEAN_H_
#define AZAH_NN_OPS_MEAN_H_

#include <stdint.h>

#include "../activation.h"
#include "../data_types.h"
#include "../node.h"
#include "../unary_op.h"

namespace azah {
namespace nn {
namespace op {

template <int Rows, int Cols>
class Mean : public UnaryOp<Rows, Cols, 1, 1> {
 public:
  Mean(const Mean&) = delete;
  Mean& operator=(const Mean&) = delete;

  Mean(Node<Rows, Cols>& input) : UnaryOp<Rows, Cols, 1, 1>(input) {}

 private:
  void ComputeOutput(uint32_t cycle) override {
    const auto& x = this->input_.Output(cycle);
    this->cached_output_ = Matrix<1, 1>::Constant(x.mean());
  }

  void UnaryBackprop(uint32_t cycle, const MatrixRef<1, 1>& output_dx) override {
    this->input_.Backprop(
        cycle, 
        Matrix<Rows, Cols>::Constant(1.0 / static_cast<float>(Rows * Cols))
            * output_dx.value());
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_OPS_MEAN_H_
