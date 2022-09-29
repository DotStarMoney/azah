#ifndef AZAH_NN_DATA_TYPES_H_
#define AZAH_NN_DATA_TYPES_H_

#include "Eigen/Dense"

namespace azah {
namespace nn {

template <int Rows, int Cols>
using Matrix = Eigen::Matrix<float, Rows, Cols>;

template <int Cols>
using RowVector = Eigen::Matrix<float, 1, Cols>;

template <int Cols>
using ColVector = Eigen::Matrix<float, Cols, 1>;

template <int Cols>
using ColVectorRef = Eigen::MatrixBase<ColVector<Cols>>;

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_DATA_TYPES_H_
