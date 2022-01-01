#ifndef AZAH_NN_CORE_H_
#define AZAH_NN_CORE_H_

#include <functional>
#include <vector>

#include "Eigen/Dense"
#include "util/random.h"

namespace azah {
namespace nn {

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Matrix<float, 1, Eigen::Dynamic> RowVector;
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> ColVector;

typedef std::vector<Matrix> BatchMatrix;
typedef std::vector<RowVector> BatchRowVector;
typedef std::vector<ColVector> BatchColVector;

typedef std::function<float(void)> ValueInitializer;

constexpr float OnesInitializer() { return 1; }
constexpr float ZerosInitializer() { return 0; }

inline ValueInitializer CreateUniformRandomInitializer(float min, float max) { 
  float delta = max - min;
  return [min, delta]() { 
        return min + static_cast<float>(util::rndd()) * delta; 
      };
}

inline BatchMatrix CreateBatchMatrix(
    int batch_n, int rows_n, int cols_n, 
    ValueInitializer initializer_fn = ZerosInitializer) {
  BatchMatrix batch_matrix(batch_n, Matrix(rows_n, cols_n));
  for (auto& matrix : batch_matrix) {
    for (int i = 0; i < matrix.size(); ++i) {
      matrix.data()[i] = initializer_fn();
    }
  }
  return batch_matrix;
}

inline BatchRowVector CreateBatchRowVector(
    int batch_n, int size_n, 
    ValueInitializer initializer_fn = ZerosInitializer) {
  BatchRowVector batch_row_vector(batch_n, RowVector(1, size_n));
  for (auto& row_vector : batch_row_vector) {
    for (int i = 0; i < row_vector.size(); ++i) {
      row_vector.data()[i] = initializer_fn();
    }
  }
  return batch_row_vector;
}

inline BatchColVector CreateBatchColVector(
    int batch_n, int size_n, 
    ValueInitializer initializer_fn = ZerosInitializer) {
  BatchColVector batch_col_vector(batch_n, ColVector(1, size_n));
  for (auto& col_vector : batch_col_vector) {
    for (int i = 0; i < col_vector.size(); ++i) {
      col_vector.data()[i] = initializer_fn();
    }
  }
  return batch_col_vector;
}

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_CORE_H_
