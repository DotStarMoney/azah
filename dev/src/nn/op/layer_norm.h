#ifndef AZAH_NN_LAYER_NORM_OP_H_
#define AZAH_NN_LAYER_NORM_OP_H_

#include <stdint.h>

#include "../binary_op.h"
#include "../data_types.h"
#include "../node.h"
#include "../op.h"
#include "../unary_op.h"
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
        output_dx.rowwise() - output_dx.colwise().mean()));
  }
};

template <int Rows, int Cols>
class SquareColMean : public UnaryOp<Rows, Cols, 1, Cols> {
 public:
  SquareColMean(const SquareMean&) = delete;
  SquareColMean& operator=(const SquareMean&) = delete;

  SquareColMean(Node<Rows, Cols>& input) : UnaryOp<Rows, Cols, 1, Cols>(input) {}

 private:
  void ComputeOutput(uint32_t cycle) override {
    const auto& x = this->input_.Output(cycle);
    this->cached_output_ = x.square().colwise().mean();
  }

  void UnaryBackprop(uint32_t cycle, 
                     const MatrixRef<1, Cols>& output_dx) override {
    const auto& x = this->input_.Output(cycle);
    this->input_.Backprop(
        cycle,
        (output_dx 
            * (2.0f / static_cast<float>(Rows))).colwise().replicate<Rows>());
  }
};

static constexpr float kEpsilon = 1e-3;

template <int Rows, int Cols>
class ColBroadcastScalarInvSqrt : public BinaryOp<Rows, Cols, 1, Cols, Rows, Cols> {
 public:
  ColBroadcastScalarInvSqrt(const ColBroadcastScalarInvSqrt&) = delete;
  ColBroadcastScalarInvSqrt& operator=(const ColBroadcastScalarInvSqrt&) = delete;

  ColBroadcastScalarInvSqrt(Node<Rows, Cols>& input_a, Node<1, 1>& input_b)
      : BinaryOp<Rows, Cols, 1, Cols, Rows, Cols>(input_a, input_b) {}

  void Backprop(uint32_t cycle, const MatrixRef<Rows, Cols>& output_dx) override {
    /*
    auto recip_inv =
      (this->input_b_.Output(cycle).array() + kEpsilon).inverse().value();
    auto recip_inv_sqrt = std::sqrt(recip_inv);

    if (!this->input_a_.constant) {
      this->input_a_.Backprop(cycle,
        (output_dx.array() * recip_inv_sqrt).matrix());
    }
    if (!this->input_b_.constant) {
      auto num = this->input_a_.Output(cycle).cwiseProduct(output_dx).array();
      this->input_b_.Backprop(cycle, Matrix<1, 1>::Constant(
        -num.sum() * 0.5 * recip_inv * recip_inv_sqrt));
    }
    */
  }

 private:
  void ComputeOutput(uint32_t cycle) override {
    const auto& x = this->input_a_.Output(cycle);
    const auto& recip = this->input_b_.Output(cycle);
    this->cached_output_ = (x.array() / 
        (recip + kEpsilon).cwiseSqrt().colwise().replicate<Rows>().array())
            .matrix();
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
class LayerNorm : public Op<Rows, Cols> {
 public:
  LayerNorm(const LayerNorm&) = delete;
  LayerNorm& operator=(const LayerNorm&) = delete;

  LayerNorm(Node<Rows, Cols>& input, Node<Rows, 1>& beta, 
            Node<Rows, 1>& gamma)
      : Op<Rows, Cols>(input.constant & beta.constant & gamma.constant),
        debias_op_(input),
        debias_fork_op_(debias_op_, 2),
        square_mean_op_(debias_fork_op_),
        scalar_inv_sqrt_op_(debias_fork_op_, square_mean_op_),
        fmadd_op_(scalar_inv_sqrt_op_, gamma, beta) {}

  void Backprop(uint32_t cycle, const MatrixRef<Rows, Cols>& output_dx) override {
    return this->fmadd_op_.Backprop(cycle, output_dx);
  }

  const Matrix<Rows, Cols>& Output(uint32_t cycle) override {
    return this->fmadd_op_.Output(cycle);
  }

 private:
  internal::Debias<Rows, Cols> debias_op_;
  Fork<Rows, Cols> debias_fork_op_;
  internal::SquareColMean<Rows, Cols> square_mean_op_;
  internal::ColBroadcastScalarInvSqrt<Rows, Cols> scalar_inv_sqrt_op_;
  internal::ColBroadcastFMAdd<Rows, Cols> fmadd_op_;

  void ComputeOutput(uint32_t cycle) override {
    LOG(FATAL) << "compute_output unimplemented for LayerNorm.";
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_LAYER_NORM_OP_H_
