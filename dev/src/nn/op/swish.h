#ifndef AZAH_NN_OPS_SWISH_H_
#define AZAH_NN_OPS_SWISH_H_

#include <stdint.h>

#include "../activation.h"
#include "../data_types.h"
#include "../op.h"
#include "../unary_op.h"

namespace azah {
namespace nn {
namespace op {

template <int InputDepth, int OutputDepth>
class Swish : public UnaryOp<InputDepth, OutputDepth> {
 public:
  Swish(const Swish&) = delete;
  Swish& operator=(const Swish&) = delete;

  Swish(Op<InputDepth>& input) : UnaryOp(input) {}

  void backprop(int32_t cycle, ColVectorRef<OutputDepth> output_dx) {
    ColVector<OutputDepth> x = this->input_.output(cycle);
    for (uint32_t i = 0; i < x.size(); ++i) {
      auto element = x.data() + i;
      *element = FastSwishD(*element);
    }
    this->input_.backprop(cycle, x.cwiseProduct(output_dx));
  }

 protected:
  void compute_output(uint32_t cycle) {
    auto x = this->input_.output(cycle);
    for (uint32_t i = 0; i < x.size(); ++i) {
      *(this->cached_output_.data() + i) = FastSwish(*(x.data() + i));
    }
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_OPS_SWISH_H_
