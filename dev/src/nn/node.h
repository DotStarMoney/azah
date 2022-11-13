#ifndef AZAH_NN_NODE_H_
#define AZAH_NN_NODE_H_

#include <stdint.h>

#include "data_types.h"
#include "node_base.h"

namespace azah {
namespace nn {

template <int OutputRows, int OutputCols>
class Node : public NodeBase {
 public:
  Node(const Node&) = delete;
  Node& operator=(const Node&) = delete;

  ConstDynamicMatrixRef OutputBase(uint32_t cycle) override {
    return Output(cycle);
  }

  void BackpropBase(uint32_t cycle) override {
    Backprop(cycle, Matrix<OutputRows, OutputCols>::Constant(1));
  }

  virtual const Matrix<OutputRows, OutputCols>& Output(uint32_t cycle) = 0;
  virtual void Backprop(uint32_t cycle, 
                        const MatrixRef<OutputRows, OutputCols>& output_dx) = 0;

  const bool constant;

 protected:
  Node(bool constant) : constant(constant) {}
};

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_NODE_H_
