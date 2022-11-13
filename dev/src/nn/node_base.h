#ifndef AZAH_NN_NODE_BASE_H_
#define AZAH_NN_NODE_BASE_H_

#include <stdint.h>

#include "data_types.h"

namespace azah {
namespace nn {

class NodeBase {
 public:
  NodeBase(const NodeBase&) = delete;
  NodeBase& operator=(const NodeBase&) = delete;

  virtual ConstDynamicMatrixRef OutputBase(uint32_t cycle) = 0;
  virtual void BackpropBase(uint32_t cycle) = 0;

 protected:
  NodeBase() {}
};

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_NODE_BASE_H_
