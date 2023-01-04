#ifndef AZAH_NN_LAYER_NORM_OP_H_
#define AZAH_NN_LAYER_NORM_OP_H_

#include <stdint.h>

#include "../binary_op.h"
#include "../data_types.h"
#include "../init.h"
#include "../node.h"
#include "../op.h"
#include "../unary_op.h"
#include "../variable.h"
#include "../variable_base.h"
#include "fork.h"
#include "glog/logging.h"

namespace azah {
namespace nn {
namespace op {
namespace internal {

template <int Rows, int Cols>
class Debias : public UnaryOp<Rows, Cols, Rows, Cols> {
 public:
  Debias(const Debias&) = delete;
  Debias& operator=(const Debias&) = delete;

  Debias(Node<Rows, Cols>& input) : UnaryOp<Rows, Cols, Rows, Cols>(input) {}

 private:
  void ComputeOutput(uint32_t cycle) override {
    const auto& x = this->input_.Output(cycle);
    this->cached_output_ = x.rowwise() - x.colwise().mean();
  }

  void UnaryBackprop(uint32_t cycle,
                     const MatrixRef<Rows, Cols>& output_dx) override {
    this->input_.Backprop(
        cycle,
        output_dx.rowwise() - output_dx.colwise().mean());
  }
};

template <int Rows, int Cols>
class SquareColMean : public UnaryOp<Rows, Cols, 1, Cols> {
 public:
  SquareColMean(const SquareColMean&) = delete;
  SquareColMean& operator=(const SquareColMean&) = delete;

  SquareColMean(Node<Rows, Cols>& input) 
      : UnaryOp<Rows, Cols, 1, Cols>(input) {}

 private:
  void ComputeOutput(uint32_t cycle) override {
    const auto& x = this->input_.Output(cycle);
    this->cached_output_ = x.array().square().matrix().colwise().mean();
  }

  void UnaryBackprop(uint32_t cycle, 
                     const MatrixRef<1, Cols>& output_dx) override {
    const auto& x = this->input_.Output(cycle);
    this->input_.Backprop(
        cycle,
        x.cwiseProduct((output_dx
            * (2.0f / static_cast<float>(Rows))).colwise().replicate<Rows>()));
  }
};

static constexpr float kEpsilon = 1e-3;

template <int Rows, int Cols>
class ColBroadcastInvSqrt 
    : public BinaryOp<Rows, Cols, 1, Cols, Rows, Cols> {
 public:
  ColBroadcastInvSqrt(const ColBroadcastInvSqrt&) = delete;
  ColBroadcastInvSqrt& operator=(const ColBroadcastInvSqrt&) = delete;

  ColBroadcastInvSqrt(Node<Rows, Cols>& input_a, Node<1, Cols>& input_b)
      : BinaryOp<Rows, Cols, 1, Cols, Rows, Cols>(input_a, input_b) {}

  void Backprop(uint32_t cycle, 
                const MatrixRef<Rows, Cols>& output_dx) override {
    auto recip_inv = 
        (this->input_b_.Output(cycle).array() + kEpsilon).inverse().matrix();
    auto recip_inv_sqrt = recip_inv.cwiseSqrt();
    if (!this->input_a_.constant) {
      this->input_a_.Backprop(
          cycle,
          output_dx.cwiseProduct(recip_inv_sqrt.colwise().replicate<Rows>()));
    }
    if (!this->input_b_.constant) {
      auto num = 0.5f * -this->input_a_.Output(cycle).cwiseProduct(output_dx);
      this->input_b_.Backprop(
          cycle,
          num.colwise().sum()
              .cwiseProduct(recip_inv).cwiseProduct(recip_inv_sqrt));
    }
  }

 private:
  void ComputeOutput(uint32_t cycle) override {
    const auto& x = this->input_a_.Output(cycle);
    const auto& recip = this->input_b_.Output(cycle);
    this->cached_output_ = (x.array() / 
        (recip.array() + kEpsilon).matrix().cwiseSqrt()
            .colwise().replicate<Rows>().array()).matrix();
  }
};

template <int Rows, int Cols>
class ColBroadcastFMAdd : public Op<Rows, Cols> {
 public:
  ColBroadcastFMAdd(const ColBroadcastFMAdd&) = delete;
  ColBroadcastFMAdd& operator=(const ColBroadcastFMAdd&) = delete;

  ColBroadcastFMAdd(Node<Rows, Cols>& input, Node<Rows, 1>& m, Node<Rows, 1>& b)
      : Op<Rows, Cols>(input.constant& m.constant& b.constant),
        input_(input),
        m_(m),
        b_(b) {}

  void Backprop(uint32_t cycle,
                const MatrixRef<Rows, Cols>& output_dx) override {
    if (!this->input_.constant) {
      const auto& m = this->m_.Output(cycle);
      this->input_.Backprop(cycle, output_dx.cwiseProduct(
          m.rowwise().replicate<Cols>()));
    }
    if (!this->m_.constant) {
      const auto& x = this->input_.Output(cycle);
      this->m_.Backprop(cycle, output_dx.cwiseProduct(x).rowwise().sum());
    }
    if (!this->b_.constant) {
      this->b_.Backprop(cycle, output_dx.rowwise().sum());
    }
  }

 private:
  Node<Rows, Cols>& input_;
  Node<Rows, 1>& m_;
  Node<Rows, 1>& b_;

  void ComputeOutput(uint32_t cycle) override {
    const auto& x = this->input_.Output(cycle);
    const auto& m = this->m_.Output(cycle);
    const auto& b = this->b_.Output(cycle);
    this->cached_output_ = (
        x.array() 
            * m.rowwise().replicate<Cols>().array() 
                + b.rowwise().replicate<Cols>().array()).matrix();
  }
};

}  // namespace internal

template <int Rows, int Cols>
class LayerNorm : public Op<Rows, Cols, 2> {
 public:
  LayerNorm(const LayerNorm&) = delete;
  LayerNorm& operator=(const LayerNorm&) = delete;

  LayerNorm(Node<Rows, Cols>& input)
      : Op<Rows, Cols, 2>(input.constant),
        debias_op_(input),
        debias_fork_op_(debias_op_, 2),
        square_mean_op_(debias_fork_op_),
        inv_sqrt_op_(debias_fork_op_, square_mean_op_),
        beta_(init::Zeros<Rows, 1>()),
        gamma_(init::Ones<Rows, 1>()),
        fmadd_op_(inv_sqrt_op_, gamma_, beta_),
        variables_{&gamma_, &beta_} {}

  void Backprop(uint32_t cycle, 
                const MatrixRef<Rows, Cols>& output_dx) override {
    this->fmadd_op_.Backprop(cycle, output_dx);
  }

  const Matrix<Rows, Cols>& Output(uint32_t cycle) override {
    return this->fmadd_op_.Output(cycle);
  }

  const std::array<VariableBase*, 2>& variables() const override {
    return variables_;
  }

 private:
  internal::Debias<Rows, Cols> debias_op_;
  Fork<Rows, Cols> debias_fork_op_;
  internal::SquareColMean<Rows, Cols> square_mean_op_;
  internal::ColBroadcastInvSqrt<Rows, Cols> inv_sqrt_op_;
  internal::ColBroadcastFMAdd<Rows, Cols> fmadd_op_;

  Variable<Rows, 1> beta_;
  Variable<Rows, 1> gamma_;
  const std::array<VariableBase*, 2> variables_;

  void ComputeOutput(uint32_t cycle) override {
    LOG(FATAL) << "compute_output unimplemented for LayerNorm.";
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_LAYER_NORM_OP_H_
