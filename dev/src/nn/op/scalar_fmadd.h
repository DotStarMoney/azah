#ifndef AZAH_NN_SCALAR_FMADD_OP_H_
#define AZAH_NN_SCALAR_FMADD_OP_H_

#include <stdint.h>

#include "../data_types.h"
#include "../node.h"
#include "../op.h"

namespace azah {
namespace nn {
namespace op {

template <int Rows, int Cols>
class ScalarFMAdd : public Op<Rows, Cols> {
 public:
  ScalarFMAdd(const ScalarFMAdd&) = delete;
  ScalarFMAdd& operator=(const ScalarFMAdd&) = delete;

  ScalarFMAdd(Node<Rows, Cols>& input, Node<1, 1>& m, Node<1, 1>& b)
      : Op<Rows, Cols>(input.constant & m.constant & b.constant),
        input_(input),
        m_(m),
        b_(b) {}

  void backprop(
      uint32_t cycle,
      const MatrixRef<Rows, Cols>& output_dx = 
          Matrix<Rows, Cols>::Constant(1)) override {
    if (!this->input_.constant) {
      auto m = this->m_.output(cycle);
      this->input_.backprop(cycle, (output_dx.array() * m.value()).matrix());
    }
    if (!this->m_.constant) {
      auto x = this->input_.output(cycle);
      this->m_.backprop(cycle, 
                        Matrix<1, 1>::Constant(output_dx.cwiseProduct(x).sum()));
    }
    if (!this->b_.constant) {
      this->b_.backprop(cycle, Matrix<1, 1>::Constant(output_dx.sum()));
    }
  }

 private:
  Node<Rows, Cols>& input_;
  Node<1, 1>& m_;
  Node<1, 1>& b_;

  void compute_output(uint32_t cycle) override {
    auto x = this->input_.output(cycle);
    auto m = this->m_.output(cycle);
    auto b = this->b_.output(cycle);
    this->cached_output_ = (x.array() * m.value() + b.value()).matrix();
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_SCALAR_FMADD_OP_H_
