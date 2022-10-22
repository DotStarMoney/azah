#ifndef AZAH_NN_OPS_SOFTMAX_H_
#define AZAH_NN_OPS_SOFTMAX_H_

#include <stdint.h>

#include "../data_types.h"
#include "../node.h"
#include "../unary_op.h"
#include "glog/logging.h"

namespace azah {
namespace nn {
namespace op {

template <int Rows, int Cols>
class Softmax : public UnaryOp<Rows, Cols, Rows, Cols> {
 public:
  Softmax(const Softmax&) = delete;
  Softmax& operator=(const Softmax&) = delete;

  Softmax(Node<Rows, Cols>& input) : UnaryOp<Rows, Cols, Rows, Cols>(input) {}

  void unary_backprop(uint32_t cycle, 
      const MatrixRef<Rows, Cols>& output_dx) override {
    LOG(FATAL) << "Backprop through Softmax is unstable. Please use "
                  "SoftmaxCrossEnt instead.";
  }

 private:
  void compute_output(uint32_t cycle) override {
    auto x = this->input_.output(cycle);
    auto x_exp = (x.array() - x.maxCoeff()).exp();
    this->cached_output_ = x_exp / (x_exp.sum() + kEpsilon);
  }

  static constexpr float kEpsilon = 1e-5;
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_OPS_SOFTMAX_H_
