#ifndef AZAH_NN_OP_H_
#define AZAH_NN_OP_H_

#include <stdint.h>

#include "data_types.h"
#include "node.h"

namespace azah {
namespace nn {

template <int OutputRows, int OutputCols>
class Op : public Node<OutputRows, OutputCols> {
 public:
  Op(const Op&) = delete;
  Op& operator=(const Op&) = delete;

  const Matrix<OutputRows, OutputCols>& output(uint32_t cycle) {
    if (cycle != cached_cycle_) {
      compute_output(cycle);
      cached_cycle_ = cycle;
    }   
    return cached_output_;
  }

 protected:
  Op(bool constant) : Node<OutputRows, OutputCols>(constant),
                      cached_output_(Matrix<OutputRows, OutputCols>::Zero()),
                      cached_cycle_(-1) {}

  virtual void compute_output(uint32_t cycle) = 0;

  Matrix<OutputRows, OutputCols> cached_output_;
 
 private:
  uint32_t cached_cycle_;
};

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_UNARY_OP_H_
