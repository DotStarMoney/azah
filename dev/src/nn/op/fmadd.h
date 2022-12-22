#ifndef AZAH_NN_FMADD_OP_H_
#define AZAH_NN_FMADD_OP_H_

#include <stdint.h>

#include "../data_types.h"
#include "../node.h"
#include "../op.h"

namespace azah {
namespace nn {
namespace op {

template <int Rows, int Cols>
class FMAdd : public Op<Rows, Cols> {
 public:
  FMAdd(const FMAdd&) = delete;
  FMAdd& operator=(const FMAdd&) = delete;

  FMAdd(Node<Rows, Cols>& input, Node<Rows, Cols>& m, Node<Rows, Cols>& b)
      : Op<Rows, Cols>(input.constant & m.constant & b.constant),
        input_(input),
        m_(m),
        b_(b) {}

  void Backprop(uint32_t cycle,
                const MatrixRef<Rows, Cols>& output_dx) override {
    if (!this->input_.constant) {
      const auto& m = this->m_.Output(cycle);
      this->input_.Backprop(cycle, output_dx.cwiseProduct(m));
    }
    if (!this->m_.constant) {
      const auto& x = this->input_.Output(cycle);
      this->m_.Backprop(cycle, output_dx.cwiseProduct(x));
    }
    if (!this->b_.constant) {
      this->b_.Backprop(cycle, output_dx);
    }
  }

 private:
  Node<Rows, Cols>& input_;
  Node<Rows, Cols>& m_;
  Node<Rows, Cols>& b_;

  void ComputeOutput(uint32_t cycle) override {
    const auto& x = this->input_.Output(cycle);
    const auto& m = this->m_.Output(cycle);
    const auto& b = this->b_.Output(cycle);
    this->cached_output_ = (x.array() * m.array() + b.array()).matrix();
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_FMADD_OP_H_
