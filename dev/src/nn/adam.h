#ifndef AZAH_NN_ADAM_H_
#define AZAH_NN_ADAM_H_

#include <stdint.h>

#include <vector>

#include "data_types.h"
#include "network.h"
#include "sgd_optimizer.h"

namespace azah {
namespace nn {

class Adam : public SGDOptimizer {
public:
  Adam(const Adam&) = delete;
  Adam& operator=(const Adam&) = delete;

  Adam(const Network& src, float beta1, float beta2);

  void Update(
      float lr, 
      const std::vector<uint32_t>& variables_i, 
      const std::vector<DynamicMatrix>& grads, 
      Network& dest) override;

 private:
  const float beta1_;
  const float beta2_;

  std::vector<DynamicMatrix> m1_;
  std::vector<DynamicMatrix> m2_;
  std::vector<uint32_t> updates_;
};

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_ADAM_H_
