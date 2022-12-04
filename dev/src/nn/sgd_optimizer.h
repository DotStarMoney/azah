#ifndef AZAH_NN_SGD_OPTIMIZER_H_
#define AZAH_NN_SGD_OPTIMIZER_H_

#include <stdint.h>

#include <vector>

#include "data_types.h"
#include "network.h"

namespace azah {
namespace nn {

class SGDOptimizer {
 public:
  SGDOptimizer(const SGDOptimizer&) = delete;
  SGDOptimizer& operator=(const SGDOptimizer&) = delete;

  virtual void Update(
      float lr, 
      const std::vector<uint32_t>& variables_i, 
      const std::vector<DynamicMatrix>& grads, 
      Network& dest) = 0;

 protected:
  SGDOptimizer() {}
};

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_SGD_OPTIMIZER_H_
