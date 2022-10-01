#ifndef AZAH_NN_ACTIVATION_H_
#define AZAH_NN_ACTIVATION_H_

namespace azah {
namespace nn {

float FastSwish(float x);
float FastSwishD(float x);

float FastSigmoid(float x);
float FastSigmoidD(float x);

float FastTanH(float x);
float FastTanHD(float x);

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_ACTIVATION_H_
