#ifndef AZAH_NN_NODE_H_
#define AZAH_NN_NODE_H_

#include <stdint.h>

#include "data_types.h"

namespace azah {
namespace nn {

template <int OutputRows, int OutputCols>
class Node {
 public:
  Node(const Node&) = delete;
  Node& operator=(const Node&) = delete;

  virtual const Matrix<OutputRows, OutputCols>& output(uint32_t cycle) = 0;
  virtual void backprop(uint32_t cycle, 
                        const MatrixRef<OutputRows, OutputCols>& output_dx) = 0;

  const bool constant;

 protected:
  Node(bool constant) : constant(constant) {}
};

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_NODE_H_
