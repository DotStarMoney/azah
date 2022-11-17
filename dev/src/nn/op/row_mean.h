#ifndef AZAH_NN_OPS_ROW_MEAN_H_
#define AZAH_NN_OPS_ROW_MEAN_H_

#include <stdint.h>

#include "../data_types.h"
#include "../node.h"
#include "../unary_op.h"

namespace azah {
namespace nn {
namespace op {

template <int Rows, int Cols>
class RowMean : public UnaryOp<Rows, Cols, Rows, 1> {
 public:
  RowMean(const RowMean&) = delete;
  RowMean& operator=(const RowMean&) = delete;

  RowMean(Node<Rows, Cols>& input) : UnaryOp<Rows, Cols, Rows, 1>(input) {}

 private:
  void ComputeOutput(uint32_t cycle) override {
    auto x = this->input_.Output(cycle);
    this->cached_output_ = x.rowwise().mean();
  }

  void UnaryBackprop(uint32_t cycle, 
                     const MatrixRef<Rows, 1>& output_dx) override {
    this->input_.Backprop(
        cycle,
        (output_dx / static_cast<float>(Cols)).colwise().replicate<1, Cols>());
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_OPS_ROW_MEAN_H_
