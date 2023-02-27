#ifndef AZAH_NN_INIT_H_
#define AZAH_NN_INIT_H_

#include <math.h>

#include "data_types.h"

namespace azah {
namespace nn {
namespace init {

template <int Rows, int Cols>
inline Matrix<Rows, Cols> Zeros() {
  return Matrix<Rows, Cols>::Zero();
}

template <int Rows, int Cols>
inline Matrix<Rows, Cols> Ones() {
  return Matrix<Rows, Cols>::Constant(1);
}

template <int Rows, int Cols, int InRows = Rows, int InCols = Cols>
inline Matrix<Rows, Cols> GlorotUniform() {
  float limit = std::sqrt(6.0 / (InRows + InCols));
  return Matrix<Rows, Cols>::Random() * limit;
}

}  // namespace init
}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_INIT_H_
