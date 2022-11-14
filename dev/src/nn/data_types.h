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
using MatrixRef = Eigen::Ref<const Matrix<Rows, Cols>>;

using DynamicMatrix = Eigen::MatrixX<float>;

using DynamicMatrixRef = Eigen::Ref<DynamicMatrix>;
using ConstDynamicMatrixRef = Eigen::Ref<const DynamicMatrix>;

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_DATA_TYPES_H_
