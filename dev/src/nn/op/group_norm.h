#ifndef AZAH_NN_GROUP_NORM_H_
#define AZAH_NN_GROUP_NORM_H_

#include <stdint.h>

#include "../data_types.h"
#include "../node.h"
#include "../op.h"

namespace azah {
namespace nn {
namespace op {
namespace {

constexpr float kEpsilon = 0.00001;

}  // namespace

template <int Groups, int Rows, int Cols>
class GroupNorm : public Op<Rows, Cols> {
  static_assert((Rows % Groups) == 0,
      "Groups must divide the input rows evenly.");

 public:
  GroupNorm(const GroupNorm&) = delete;
  GroupNorm& operator=(const GroupNorm&) = delete;
  GroupNorm(Node<Rows, Cols>& input, Node<Groups, 1>& gamma,
            Node<Groups, 1>& beta) :
      Op<Rows, Cols>(input.constant & gamma.constant & beta.constant),
      input_(input),
      gamma_(gamma),
      beta_(beta) {}

  void backprop(
      uint32_t cycle,
      const MatrixRef<Rows, Cols>& output_dx = Matrix<Rows, Cols>::Constant(1)) {
    //
    // Fill me in!
    //
  }

 protected:
  void compute_output(uint32_t cycle) {
    auto x = this->input_.output(cycle);
    auto gamma = this->gamma_.output(cycle);
    auto beta = this->beta_.output(cycle);

    for (int g = 0; g < Groups; ++g) {
      Eigen::Map<Matrix<Rows / Groups, Cols>> group_x(
          x.data() + g * Rows / Groups * Cols);

      auto mu = group_x.mean();
      Matrix<Rows / Groups, Cols> debiased = group_x - mu;
      auto sigma = debiased.square().mean();
      auto norm = debiased / (sigma + kEpsilon).sqrt();

      this->cached_output_.middleRows(g * Rows / Groups, Rows / Groups) =
          norm * gamma(g) + beta(g);
    }
  }

 private:
  Node<Rows, Cols>& input_;
  Node<Groups, 1>& gamma_;
  Node<Groups, 1>& beta_;
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_GROUP_NORM_H_
