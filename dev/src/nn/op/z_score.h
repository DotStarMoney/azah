#ifndef AZAH_NN_Z_SCORE_OP_H_
#define AZAH_NN_Z_SCORE_OP_H_

#include <stdint.h>

#include "../data_types.h"
#include "../node.h"
#include "../op.h"

namespace azah {
namespace nn {
namespace op {

template <int Rows, int Cols>
class ZScore : public Op<Rows, Cols> {
 public:
  ZScore(const ZScore&) = delete;
  ZScore& operator=(const ZScore&) = delete;

  ZScore(Node<Rows, Cols>& input, Node<1, 1>& mean, Node<1, 1>& var)
      : Op<Rows, Cols>(input.constant & mean.constant & var.constant),
        input_(input),
        mean_(mean),
        var_(var) {}

  void backprop(
      uint32_t cycle,
      const MatrixRef<Rows, Cols>& output_dx = Matrix<Rows, Cols>::Constant(1)) {
    if (!this->input_.constant) {

    }
    if (!this->mean_.constant) {

    }
    if (!this->var_.constant) {

    }
  }

 private:
  Node<Rows, Cols>& input_;
  Node<1, 1>& mean_;
  Node<1, 1>& var_;

  void compute_output(uint32_t cycle) {
    auto x = this->input_.output(cycle);
    auto mean = this->mean_.output(cycle);
    auto var = this->var_.output(cycle);
    this->cached_output_ = (x.array() - mean_.value()) 
        / (var.array() + kEpsilon).sqrt().value();
  }

  constexpr float kEpsilon = 1e-5;
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_Z_SCORE_OP_H_
