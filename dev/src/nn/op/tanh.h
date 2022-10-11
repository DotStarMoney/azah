#ifndef AZAH_NN_OPS_TANH_H_
#define AZAH_NN_OPS_TANH_H_

#include <stdint.h>

#include "../activation.h"
#include "../data_types.h"
#include "../node.h"
#include "../unary_op.h"

namespace azah {
namespace nn {
namespace op {

template <int Rows, int Cols>
class TanH : public UnaryOp<Rows, Cols, Rows, Cols> {
 public:
  TanH(const Swish&) = delete;
  TanH& operator=(const TanH&) = delete;

  TanH(Node<Rows, Cols>& input) : UnaryOp<Rows, Cols, Rows, Cols>(input),
      cached_input_dx_(Matrix<Rows, Cols>::Zero()),
      grad_cycle_(-1) {}

  void unary_backprop(
      uint32_t cycle,
      const MatrixRef<Rows, Cols>& output_dx = Matrix<Rows, Cols>::Constant(1)) {
    if (cycle != grad_cycle_) {
      auto x = this->input_.output(cycle);
      for (uint32_t i = 0; i < x.size(); ++i) {
        *(cached_input_dx_.data() + i) = FastTanHD(*(x.data() + i));
      }
      grad_cycle_ = cycle;
    }
    this->input_.backprop(cycle, cached_input_dx_.cwiseProduct(output_dx));
  }

 protected:
  void compute_output(uint32_t cycle) {
    auto x = this->input_.output(cycle);
    for (uint32_t i = 0; i < x.size(); ++i) {
      *(this->cached_output_.data() + i) = FastTanH(*(x.data() + i));
    }
  }

 private:
  Matrix<Rows, Cols> cached_input_dx_;
  uint32_t grad_cycle_;
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_OPS_TANH_H_
