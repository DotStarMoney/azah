#ifndef AZAH_NN_OP_H_
#define AZAH_NN_OP_H_

#include <stdint.h>

#include <array>

#include "data_types.h"
#include "glog/logging.h"
#include "node.h"
#include "variable_base.h"

namespace azah {
namespace nn {

template <int OutputRows, int OutputCols, int VariablesN = 0>
class Op : public Node<OutputRows, OutputCols> {
 public:
  Op(const Op&) = delete;
  Op& operator=(const Op&) = delete;

  virtual const Matrix<OutputRows, OutputCols>& Output(
      uint32_t cycle) override {
    if (cycle != cached_cycle_) {
      ComputeOutput(cycle);
      cached_cycle_ = cycle;
    }   
    return cached_output_;
  }

  virtual const std::array<VariableBase*, VariablesN>& variables() const {
    LOG(FATAL) << "This op does not expose internal variables.";
  }

 protected:
  Op(bool constant) : Node<OutputRows, OutputCols>(constant),
                      cached_output_(Matrix<OutputRows, OutputCols>::Zero()),
                      cached_cycle_(-1) {}

  virtual void ComputeOutput(uint32_t cycle) = 0;

  Matrix<OutputRows, OutputCols> cached_output_;
 
 private:
  uint32_t cached_cycle_;
};

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_UNARY_OP_H_
