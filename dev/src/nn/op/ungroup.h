#ifndef AZAH_NN_OPS_UNGROUP_H_
#define AZAH_NN_OPS_UNGROUP_H_

#include <stdint.h>

#include "../data_types.h"
#include "../node.h"
#include "../unary_op.h"
#include "Eigen/Core"

namespace azah {
namespace nn {
namespace op {

template <int Groups, int Rows, int Cols>
class Ungroup : public UnaryOp<Rows * Cols / Groups, Groups, Rows, Cols> {
 public:
  Ungroup(const Ungroup&) = delete;
  Ungroup& operator=(const Ungroup&) = delete;

  Ungroup(Node<Rows * Cols / Groups, Groups>& input) : 
      UnaryOp<Rows * Cols / Groups, Groups, Rows, Cols>(input) {}

  void unary_backprop(uint32_t cycle, const MatrixRef<Rows, Cols>& output_dx) {
    Matrix<Cols, Rows> x_t = output_dx.transpose();
    this->input_.backprop(
        cycle, 
        Eigen::Map<Matrix<Rows * Cols / Groups, Groups>>(x_t.data()));
  }

 private:
  void compute_output(uint32_t cycle) {
    this->cached_output_ = Eigen::Map<const Matrix<Cols, Rows>>(
        this->input_.output(cycle).data()).transpose();
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_OPS_GROUP_H_
