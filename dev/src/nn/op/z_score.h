#ifndef AZAH_NN_Z_SCORE_OP_H_
#define AZAH_NN_Z_SCORE_OP_H_

#include <math.h>
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
      const MatrixRef<Rows, Cols>& output_dx = 
          Matrix<Rows, Cols>::Constant(1)) override {
    auto var_inv = (this->var_.output(cycle).array() + kEpsilon).inverse().value();
    auto stddev_inv = std::sqrt(var_inv);
    auto out_dx_sum = output_dx.sum();
    if (!this->input_.constant) {
      this->input_.backprop(cycle, (output_dx.array() * stddev_inv).matrix());
    }
    if (!this->mean_.constant) {
      this->mean_.backprop(cycle, 
                           Matrix<1, 1>::Constant(-out_dx_sum * stddev_inv));
    }
    if (!this->var_.constant) {
      auto num = this->mean_.output(cycle).array() * out_dx_sum
          - (this->input_.output(cycle).cwiseProduct(output_dx).array()).sum();
      this->var_.backprop(cycle, Matrix<1, 1>::Constant(
          num.value() * 0.5 * var_inv * stddev_inv));
    }
  }

 private:
  Node<Rows, Cols>& input_;
  Node<1, 1>& mean_;
  Node<1, 1>& var_;

  void compute_output(uint32_t cycle) override {
    auto x = this->input_.output(cycle);
    auto mean = this->mean_.output(cycle);
    auto var = this->var_.output(cycle);
    this->cached_output_ = (x.array() - mean.value()) 
        / (var.array() + kEpsilon).sqrt().value();
  }

  static constexpr float kEpsilon = 1e-5;
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_Z_SCORE_OP_H_
