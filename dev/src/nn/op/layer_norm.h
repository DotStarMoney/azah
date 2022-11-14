#ifndef AZAH_NN_LAYER_NORM_OP_H_
#define AZAH_NN_LAYER_NORM_OP_H_

#include <stdint.h>

#include "../data_types.h"
#include "../node.h"
#include "../op.h"
#include "fork.h"
#include "glog/logging.h"
#include "mean.h"
#include "scalar_fmadd.h"
#include "scalar_mse.h"
#include "z_score.h"

namespace azah {
namespace nn {
namespace op {

template <int Rows, int Cols>
class LayerNorm : public Op<Rows, Cols> {
 public:
  LayerNorm(const LayerNorm&) = delete;
  LayerNorm& operator=(const LayerNorm&) = delete;

  LayerNorm(Node<Rows, Cols>& input, Node<1, 1>& beta, Node<1, 1>& gamma)
      : Op<Rows, Cols>(input.constant & beta.constant & gamma.constant),
        input_fork_op_(input, 3),
        mean_op_(input_fork_op_),
        mean_fork_op_(mean_op_, 2),
        scalar_mse_op_(input_fork_op_, mean_fork_op_),
        z_score_op_(input_fork_op_, mean_fork_op_, scalar_mse_op_),
        scalar_fmadd_op_(z_score_op_, gamma, beta) {}

  void Backprop(uint32_t cycle, const MatrixRef<Rows, Cols>& output_dx) override {
    return this->scalar_fmadd_op_.Backprop(cycle, output_dx);
  }

  const Matrix<Rows, Cols>& Output(uint32_t cycle) override {
    return this->scalar_fmadd_op_.output(cycle);
  }

 private:
  Fork<Rows, Cols> input_fork_op_;
  Mean<Rows, Cols> mean_op_;
  Fork<1, 1> mean_fork_op_;
  ScalarMSE<Rows, Cols> scalar_mse_op_;
  ZScore<Rows, Cols> z_score_op_;
  ScalarFMAdd<Rows, Cols> scalar_fmadd_op_;

  void ComputeOutput(uint32_t cycle) override {
    LOG(FATAL) << "compute_output unimplemented for LayerNorm.";
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_LAYER_NORM_OP_H_
