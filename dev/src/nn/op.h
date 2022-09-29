#ifndef AZAH_NN_OP_H_
#define AZAH_NN_OP_H_

#include <stdint.h>

#include "data_types.h"
#include <Eigen/Core>

namespace azah {
namespace nn {

template <int OutputDepth>
class Op {
 public:
  Op(const Op&) = delete;
  Op& operator=(const Op&) = delete;

  ColVectorRef<OutputDepth> output(uint32_t cycle) {
    if (cycle != this->cached_cycle_) {
      output();
      this->cached_cycle_ = cycle;
    }   
    return cached_output_;
  }

  virtual void backprop(uint32_t cycle, ColVectorRef<OutputDepth> output_dx) = 0;

 protected:
  Op() : cached_output_(ColVector<OutputDepth>::Zero()), cached_cycle_(-1) {}

  virtual void compute_output(uint32_t cycle) = 0;

  ColVector<OutputDepth> cached_output_;
 
 private:
  uint32_t cached_cycle_;
};

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_UNARY_OP_H_
