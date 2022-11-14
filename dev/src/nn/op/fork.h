#ifndef AZAH_NN_OPS_FORK_H_
#define AZAH_NN_OPS_FORK_H_

#include <stdint.h>

#include "../data_types.h"
#include "../node.h"
#include "../unary_op.h"
#include "glog/logging.h"

namespace azah {
namespace nn {
namespace op {

template <int Rows, int Cols>
class Fork : public UnaryOp<Rows, Cols, Rows, Cols> {
 public:
  Fork(const Fork&) = delete;
  Fork& operator=(const Fork&) = delete;

  Fork(Node<Rows, Cols>& input, int forks_n = 2) 
      : UnaryOp<Rows, Cols, Rows, Cols>(input),
        grad_cycle_(-1),
        forks_(0),
        n_forks_(forks_n) {}

  const Matrix<Rows, Cols>& Output(uint32_t cycle) {
    return this->input_.Output(cycle);
  }
 

 private:
  Matrix<Rows, Cols> forked_grad_;
  uint32_t grad_cycle_;
  uint32_t forks_;
  const uint32_t n_forks_;

  void ComputeOutput(uint32_t cycle) override {
    LOG(FATAL) << "compute_output unimplemented for Fork.";
  }

  void UnaryBackprop(
      uint32_t cycle,
      const MatrixRef<Rows, Cols>& output_dx =
          Matrix<Rows, Cols>::Constant(1)) override {
    if (cycle != grad_cycle_) {
      forked_grad_ = output_dx;
      grad_cycle_ = cycle;
      forks_ = 1;
      return;
    }
    forked_grad_ += output_dx;
    ++forks_;
    if (forks_ == n_forks_) {
      this->input_.Backprop(cycle, forked_grad_);
    } else if (forks_ > n_forks_) {
      LOG(FATAL) << "Too many forks, did you mean to set \"forks_n\" in the "
          "constructor?";
    }
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_OPS_FORK_H_
