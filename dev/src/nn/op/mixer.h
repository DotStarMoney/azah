#ifndef AZAH_NN_MIXER_OP_H_
#define AZAH_NN_MIXER_OP_H_

#include <stdint.h>

#include <array>

#include "../data_types.h"
#include "../init.h"
#include "../node.h"
#include "../op.h"
#include "../variable.h"
#include "../variable_base.h"
#include "add.h"
#include "fork.h"
#include "layer_norm.h"
#include "matmul.h"
#include "swish.h"
#include "transpose.h"

#include "glog/logging.h"

namespace azah {
namespace nn {
namespace op {

template <int Rows, int Cols, int TokenHiddenSize, int FeatureHiddenSize>
class Mixer : public Op<Rows, Cols, 8> {
 public:
  Mixer(const Mixer&) = delete;
  Mixer& operator=(const Mixer&) = delete;

  Mixer(Node<Rows, Cols>& input)
      : Op<Rows, Cols, 8>(input.constant),
        fork_t_(input, 2),
        norm_t_(fork_t_),
        dense_t_1_k_(init::GlorotUniform<TokenHiddenSize, Cols>()),
        dense_t_1_(dense_t_1_k_, norm_t_),
        swish_t_(dense_t_1_),
        dense_t_2_k_(init::GlorotUniform<Cols, TokenHiddenSize>()),
        dense_t_2_(dense_t_2_k_, swish_t_),
        transpose_(dense_t_2_),
        res_t_(transpose_, fork_t_),
        fork_f_(res_t_, 2),
        norm_f_(fork_f_),
        dense_f_1_k_(init::GlorotUniform<FeatureHiddenSize, Rows>()),
        dense_f_1_(dense_f_1_k_, norm_f_),
        swish_f_(dense_f_1_),
        dense_f_2_k_(init::GlorotUniform<Rows, FeatureHiddenSize>()),
        dense_f_2_(dense_f_2_k_, swish_f_),
        res_f_(dense_f_2_, fork_f_),
        variables_{
            norm_t_.variables()[0], norm_t_.variables()[1], &dense_t_1_k_, 
            &dense_t_2_k_, norm_f_.variables()[0], norm_f_.variables()[1], 
            &dense_f_1_k_, &dense_f_2_k_} {}

  void Backprop(uint32_t cycle, 
                const MatrixRef<Rows, Cols>& output_dx) override {
    this->res_f_.Backprop(cycle, output_dx);
  }

  const Matrix<Rows, Cols>& Output(uint32_t cycle) override {
    return this->res_f_.Output(cycle);
  }

  const std::array<VariableBase*, 8>& variables() const override {
    return variables_;
  }

 private:
  Fork<Rows, Cols> fork_t_;
  LayerNorm<Rows, Cols> norm_t_;
  Variable<TokenHiddenSize, Cols> dense_t_1_k_;
  Matmul<TokenHiddenSize, Cols, Rows, Cols, true> dense_t_1_;
  Swish<TokenHiddenSize, Rows> swish_t_;
  Variable<Cols, TokenHiddenSize> dense_t_2_k_;
  Matmul<Cols, TokenHiddenSize, TokenHiddenSize, Rows> dense_t_2_;
  Transpose<Cols, Rows> transpose_;
  Add<Rows, Cols> res_t_;

  Fork<Rows, Cols> fork_f_;
  LayerNorm<Rows, Cols> norm_f_;
  Variable<FeatureHiddenSize, Rows> dense_f_1_k_;
  Matmul<FeatureHiddenSize, Rows, Rows, Cols> dense_f_1_;
  Swish<FeatureHiddenSize, Cols> swish_f_;
  Variable<Rows, FeatureHiddenSize> dense_f_2_k_;
  Matmul<Rows, FeatureHiddenSize, FeatureHiddenSize, Cols> dense_f_2_;
  Add<Rows, Cols> res_f_;

  const std::array<VariableBase*, 8> variables_;

  void ComputeOutput(uint32_t cycle) override {
    LOG(FATAL) << "compute_output unimplemented for Mixer.";
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_MIXER_OP_H_
