#ifndef AZAH_NN_OPS_BROADCAST_ADD_H_
#define AZAH_NN_OPS_BROADCAST_ADD_H_

#include <stdint.h>

#include "../binary_op.h"
#include "../data_types.h"
#include "../node.h"

namespace azah {
namespace nn {
namespace op {

template <int Rows, int Cols>
class BroadcastAdd : public BinaryOp<Rows, 1, Rows, Cols, Rows, Cols> {
 public:
  BroadcastAdd(const BroadcastAdd&) = delete;
  BroadcastAdd& operator=(const BroadcastAdd&) = delete;

  BroadcastAdd(Node<Rows, 1>& input_a, Node<Rows, Cols>& input_b)
      : BinaryOp<Rows, 1, Rows, Cols, Rows, Cols>(input_a, input_b) {}

  void Backprop(uint32_t cycle, 
                const MatrixRef<Rows, Cols>& output_dx) override {
    if (!this->input_a_.constant) {
      this->input_a_.Backprop(cycle, output_dx.rowwise().sum());
    }
    if (!this->input_b_.constant) {
      this->input_b_.Backprop(cycle, output_dx);
    }
  }

 protected:
  void ComputeOutput(uint32_t cycle) override {
    const auto& a = this->input_a_.Output(cycle);
    const auto& b = this->input_b_.Output(cycle);
    this->cached_output_ = a + b.colwise();
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_OPS_BROADCAST_ADD_H_
