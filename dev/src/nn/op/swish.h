#ifndef AZAH_NN_OPS_SWISH_H_
#define AZAH_NN_OPS_SWISH_H_

#include <stdint.h>

#include "../activation.h"
#include "../data_types.h"
#include "../node.h"
#include "../unary_op.h"

namespace azah {
namespace nn {
namespace op {

template <int Rows, int Cols>
class Swish : public UnaryOp<Rows, Cols, Rows, Cols> {
 public:
  Swish(const Swish&) = delete;
  Swish& operator=(const Swish&) = delete;

  Swish(Node<Rows, Cols>& input) : UnaryOp<Rows, Cols, Rows, Cols>(input) {}

  void unary_backprop(uint32_t cycle, const MatrixRef<Rows, Cols>& output_dx) {
    Matrix<Rows, Cols> x = this->input_.output(cycle);
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
