#ifndef AZAH_NN_UNARY_OP_H_
#define AZAH_NN_UNARY_OP_H_

#include <stdint.h>

#include "data_types.h"
#include "node.h"
#include "op.h"

namespace azah {
namespace nn {

template <int InputRows, int InputCols, int OutputRows, int OutputCols>
class UnaryOp : public Op<OutputRows, OutputCols> {
 public:
  UnaryOp(const UnaryOp&) = delete;
  UnaryOp& operator=(const UnaryOp&) = delete;

  void Backprop(
      uint32_t cycle,
      const MatrixRef<OutputRows, OutputCols>& output_dx =
          Matrix<OutputRows, OutputCols>::Constant(1)) override {
    if (input_.constant) return;
    UnaryBackprop(cycle, output_dx);
  }

 protected:
  UnaryOp(Node<InputRows, InputCols>& input) 
      : Op<OutputRows, OutputCols>(input.constant), input_(input) {}

  Node<InputRows, InputCols>& input_;

 private:
  virtual void UnaryBackprop(
      uint32_t cycle, const MatrixRef<OutputRows, OutputCols>& output_dx) = 0;
};

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_UNARY_OP_H_
