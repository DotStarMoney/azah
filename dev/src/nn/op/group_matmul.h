#ifndef AZAH_NN_OPS_GROUP_MATMUL_H_
#define AZAH_NN_OPS_GROUP_MATMUL_H_

#include <stdint.h>

#include "../binary_op.h"
#include "../data_types.h"
#include "../node.h"

namespace azah {
namespace nn {
namespace op {

template <int Groups, int InputRowsA, int InputColsA, int InputRowsB, 
          int InputColsB>
class GroupMatmul : public BinaryOp<InputRowsA, InputColsA, InputRowsB, InputColsB,
                                    InputRowsA * Groups, InputColsB> {
  static_assert((InputColsA % Groups == 0) && (InputRowsB % Groups == 0),
                "Groups must divide the columns of A and rows of B evenly.");
 public:
  GroupMatmul(const GroupMatmul&) = delete;
  GroupMatmul& operator=(const GroupMatmul&) = delete;

  GroupMatmul(Node<InputRowsA, InputColsA>& input_a,
              Node<InputRowsB, InputColsB>& input_b) :
      BinaryOp<InputRowsA, InputColsA, InputRowsB, InputColsB, InputRowsA * Groups,
               InputColsB>(input_a, input_b) {}

  void backprop(
      uint32_t cycle,
      const MatrixRef<InputRowsA * Groups, InputColsB>& output_dx = Matrix<
          InputRowsA * Groups, InputColsB>::Constant(1)) {
    if (!this->input_a_.constant) {
      Matrix<InputRowsA, InputColsA> j;
      Matrix<InputColsB, InputRowsB> b_trans =
          this->input_b_.output(cycle).transpose();
      for (int g = 0; g < Groups; ++g) {
        j.middleCols(g * InputColsA / Groups, InputColsA / Groups) =
            output_dx.middleRows(g * InputRowsA, InputRowsA) *
            b_trans.middleCols(g * InputRowsB / Groups, InputRowsB / Groups);
      }
      this->input_a_.backprop(cycle, j);
    }

    if (!this->input_b_.constant) {
      Matrix<InputRowsB, InputColsB> j;
      Matrix<InputColsA, InputRowsA> a_trans =
          this->input_a_.output(cycle).transpose();
      for (int g = 0; g < Groups; ++g) {
        j.middleRows(g * InputRowsB / Groups, InputRowsB / Groups) =
            a_trans.middleRows(g * InputColsA / Groups, InputColsA / Groups) *
            output_dx.middleRows(g * InputRowsA, InputRowsA);
      }
      this->input_b_.backprop(cycle, j);
    }
  }

 private:
  void compute_output(uint32_t cycle) {
    auto a = this->input_a_.output(cycle);
    auto b = this->input_b_.output(cycle);
    for (int g = 0; g < Groups; ++g) {
      this->cached_output_.middleRows(g * InputRowsA, InputRowsA) = 
          a.middleCols(g * InputColsA / Groups, InputColsA / Groups) * 
          b.middleRows(g * InputRowsB / Groups, InputRowsB / Groups);
    }
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_OPS_GROUP_MATMUL_H_
