#ifndef AZAH_NN_NETWORK_H_
#define AZAH_NN_NETWORK_H_

#include <stdint.h>

#include "data_types.h"

namespace azah {
namespace nn {

class Network {
 public:
  Network(const Network&) = delete;
  Network& operator=(const Network&) = delete;

  void Forward(uint32_t output_i);


 protected:
  uint32_t cycle_;
  const uint32_t outputs_n_;

  Network(uint32_t outputs_n);
};

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_NETWORK_H_
