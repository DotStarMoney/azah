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
  TanH(const TanH&) = delete;
  TanH& operator=(const TanH&) = delete;

  TanH(Node<Rows, Cols>& input) 
      : UnaryOp<Rows, Cols, Rows, Cols>(input),
        grad_cycle_(-1) {}

 private:
  Matrix<Rows, Cols> cached_input_dx_;
  uint32_t grad_cycle_;

  void ComputeOutput(uint32_t cycle) override {
    const auto& x = this->input_.Output(cycle);
    for (uint32_t i = 0; i < x.size(); ++i) {
      *(this->cached_output_.data() + i) = FastTanH(*(x.data() + i));
    }
  }

  void UnaryBackprop(uint32_t cycle,
                     const MatrixRef<Rows, Cols>& output_dx) override {
    if (cycle != grad_cycle_) {
      const auto& x = this->input_.Output(cycle);
      for (uint32_t i = 0; i < x.size(); ++i) {
        *(cached_input_dx_.data() + i) = FastTanHD(*(x.data() + i));
      }
      grad_cycle_ = cycle;
    }
    this->input_.Backprop(cycle, cached_input_dx_.cwiseProduct(output_dx));
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_OPS_TANH_H_
