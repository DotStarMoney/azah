#ifndef AZAH_NN_OPS_ADD_H_
#define AZAH_NN_OPS_ADD_H_

#include <stdint.h>

#include "../binary_op.h"
#include "../data_types.h"
#include "../node.h"

namespace azah {
namespace nn {
namespace op {

template <int Rows, int Cols>
class Add : public BinaryOp<Rows, Cols, Rows, Cols, Rows, Cols> {
 public:
  Add(const Add&) = delete;
  Add& operator=(const Add&) = delete;

  Add(Node<Rows, Cols>& input_a, Node<Rows, Cols>& input_b) :
    BinaryOp<Rows, Cols, Rows, Cols, Rows, Cols>(input_a, input_b) {}

  void backprop(uint32_t cycle, const MatrixRef<Rows, Cols>& output_dx) {
    if (!this->input_a_.constant) {
      this->input_a_.backprop(cycle, output_dx);
    }
    if (!this->input_b_.constant) {
      this->input_b_.backprop(cycle, output_dx);
    }
  }

 protected:
  void compute_output(uint32_t cycle) {
    auto a = this->input_a_.output(cycle);
    auto b = this->input_b_.output(cycle);
    this->cached_output_ = a + b;
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_OPS_ADD_H_
