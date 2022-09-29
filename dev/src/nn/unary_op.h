#ifndef AZAH_NN_UNARY_OP_H_
#define AZAH_NN_UNARY_OP_H_

#include <stdint.h>

#include "data_types.h"
#include "op.h"

namespace azah {
namespace nn {

template <int InputDepth, int OutputDepth>
class UnaryOp : public Op<OutputDepth> {
 public:
  UnaryOp(const UnaryOp&) = delete;
  UnaryOp& operator=(const UnaryOp&) = delete;

 protected:
  UnaryOp(Op<InputDepth>& input) : Op(), input_(input) {}

  Op<InputDepth>& const input_;
};

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_UNARY_OP_H_
