#ifndef AZAH_NN_DATA_TYPES_H_
#define AZAH_NN_DATA_TYPES_H_

#define _USE_MATH_DEFINES
#include <math.h>

#include "Eigen/Core"

namespace azah {
namespace nn {

template <int Rows, int Cols>
using Matrix = Eigen::Matrix<float, Rows, Cols>;

template <int Rows, int Cols>
using MatrixRef = Eigen::Ref<const Eigen::Matrix<float, Rows, Cols>>;

using DynamicMatrix = Eigen::MatrixX<float>;

using DynamicMatrixRef = Eigen::Ref<Eigen::MatrixX<float>>;
using ConstDynamicMatrixRef = Eigen::Ref<const Eigen::MatrixX<float>>;

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_DATA_TYPES_H_
