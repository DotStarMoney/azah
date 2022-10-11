#ifndef AZAH_NN_OPS_GROUP_MATMUL_H_
#define AZAH_NN_OPS_GROUP_MATMUL_H_

#include <stdint.h>

#include "../binary_op.h"
#include "../data_types.h"
#include "../node.h"
#include "Eigen/Core"

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
      for (int g = 0; g < Groups; ++g) {
        Eigen::Map<const Matrix<InputRowsA, InputColsB>> output_dx_group(
            output_dx.data() + g * InputRowsA * InputColsB);
        Eigen::Map<
            const Matrix<InputColsB, InputRowsB / Groups>,
            0,
            Eigen::Stride<InputRowsB, 1>> input_b_t_group(
                this->input_b_.output(cycle).transpose().data()
                    + g * InputRowsB / Groups);
        j.middleCols(g * InputColsA / Groups, InputColsA / Groups) = 
            output_dx_group * input_b_t_group;
      }
      this->input_a_.backprop(cycle, j);
    }
    
    if (!this->input_b_.constant) {
      Matrix<InputRowsB, InputColsB> j;
      for (int g = 0; g < Groups; ++g) {
        Eigen::Map<const Matrix<InputRowsA, InputColsB>> output_dx_group(
            output_dx.data() + g * InputRowsA * InputColsB);
        Eigen::Map<const Matrix<InputColsA / Groups, InputRowsA>> input_a_t_group(
            this->input_a_.output(cycle).transpose().data() 
                + g * InputRowsA * InputColsA / Groups);
        j.middleRows(g * InputRowsB / Groups, InputRowsB / Groups) =
            input_a_t_group * output_dx_group;
      }
      this->input_b_.backprop(cycle, j);
    }
  }

 protected:
  void compute_output(uint32_t cycle) {
    auto a = this->input_a_.output(cycle);
    auto b = this->input_b_.output(cycle);
    for (int g = 0; g < Groups; ++g) {
      Eigen::Map<
          Matrix<InputRowsA, InputColsA / Groups>, 
          0, 
          Eigen::Stride<InputColsA, 1>> group_a(
              a.data() + g * InputColsA / Groups);
      Eigen::Map<Matrix<InputRowsB / Groups, InputColsB>> group_b(
          b.data() + g * InputRowsB / Groups * InputColsB);
      this->cached_output_.middleRows(g * InputRowsA, InputRowsA) = 
          group_a * group_b;
    }
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_OPS_GROUP_MATMUL_H_
