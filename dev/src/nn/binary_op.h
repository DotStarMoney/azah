#ifndef AZAH_NN_BINARY_OP_H_
#define AZAH_NN_BINARY_OP_H_

#include <stdint.h>

#include "data_types.h"
#include "node.h"
#include "op.h"

namespace azah {
namespace nn {

template <int InputRowsA, int InputColsA, int InputRowsB, int InputColsB, 
          int OutputRows, int OutputCols>
class BinaryOp : public Op<OutputRows, OutputCols> {
 public:
  BinaryOp(const BinaryOp&) = delete;
  BinaryOp& operator=(const BinaryOp&) = delete;

 protected:
  BinaryOp(Node<InputRowsA, InputColsA>& input_a,
           Node<InputRowsB, InputColsB>& input_b)
      : Op<OutputRows, OutputCols>(input_a.constant & input_b.constant),
        input_a_(input_a),
        input_b_(input_b) {}

  Node<InputRowsA, InputColsA>& input_a_;
  Node<InputRowsB, InputColsB>& input_b_;
};

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_BINARY_OP_H_
