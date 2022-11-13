#ifndef AZAH_NN_OPS_TRANSPOSE_H_
#define AZAH_NN_OPS_TRANSPOSE_H_

#include <stdint.h>

#include "../unary_op.h"
#include "../data_types.h"
#include "../node.h"

namespace azah {
namespace nn {
namespace op {

template <int Rows, int Cols>
class Transpose : public UnaryOp<Rows, Cols, Cols, Rows> {
 public:
  Transpose(const Transpose&) = delete;
  Transpose& operator=(const Transpose&) = delete;

  Transpose(Node<Rows, Cols>& input) : UnaryOp<Rows, Cols, Cols, Rows>(input) {}

 private:
  void ComputeOutput(uint32_t cycle) override {
    this->cached_output_ = this->input_.Output(cycle).transpose();
  }

  void UnaryBackprop(uint32_t cycle,
                     const MatrixRef<Cols, Rows>& output_dx) override {
    this->input_.Backprop(cycle, output_dx.transpose());
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_OPS_TRANSPOSE_H_
