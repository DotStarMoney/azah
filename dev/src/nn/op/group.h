#ifndef AZAH_NN_OPS_GROUP_H_
#define AZAH_NN_OPS_GROUP_H_

#include <stdint.h>

#include "../data_types.h"
#include "../node.h"
#include "../unary_op.h"
#include "Eigen/Core"

namespace azah {
namespace nn {
namespace op {

template <int Groups, int Rows, int Cols>
class Group : public UnaryOp<Rows, Cols, Rows * Cols / Groups, Groups> {
 public:
  Group(const Group&) = delete;
  Group& operator=(const Group&) = delete;

  Group(Node<Rows, Cols>& input) : UnaryOp<Rows, Cols, Rows * Cols / Groups,
                                           Groups>(input) {}

  void unary_backprop(uint32_t cycle,
      const MatrixRef<Rows * Cols / Groups, Groups>& output_dx) {
    this->input_.backprop(cycle, 
                          Eigen::Map<const Matrix<Cols, Rows>>(output_dx.data())
        .transpose());
  }

 private:
  void compute_output(uint32_t cycle) {
    Matrix<Cols, Rows> x_t = this->input_.output(cycle).transpose();
    this->cached_output_ = Eigen::Map<Matrix<Rows * Cols / Groups, Groups>>(
        x_t.data());
  }
};

}  // namespace op
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_OPS_GROUP_H_
