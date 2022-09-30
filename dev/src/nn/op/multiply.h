#ifndef AZAH_NN_OPS_MULTIPLY_H_
#define AZAH_NN_OPS_MULTIPLY_H_

#include <stdint.h>

#include "../binary_op.h"
#include "../data_types.h"
#include "../node.h"

namespace azah {
namespace nn {
namespace op {

template <int Rows, int Cols>
class Multiply : public BinaryOp<Rows, Cols, Rows, Cols, Rows, Cols> {
 public:
  Multiply(const Multiply&) = delete;
  Multiply& operator=(const Multiply&) = delete;

  Multiply(Node<Rows, Cols>& input_a, Node<Rows, Cols>& input_b) :
      BinaryOp<Rows, Cols, Rows, Cols, Rows, Cols>(input_a, input_b) {}

  void backprop(uint32_t cycle, const MatrixRef<Rows, Cols>& output_dx) {
    if (!this->input_a_.constant) {
      this->input_a_.backprop(
          cycle, this->input_b_.output(cycle).cwiseProduct(output_dx));
    }
    if (!this->input_b_.constant) {
      this->input_b_.backprop(
          cycle, this->input_a_.output(cycle).cwiseProduct(output_dx));
    }
  }

 protected:
  void compute_output(uint32_t cycle) {
    auto a = this->input_a_.output(cycle);
    auto b = this->input_b_.output(cycle);
    this->cached_output_ = a.cwiseProduct(b);
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_OPS_MULTIPLY_H_
