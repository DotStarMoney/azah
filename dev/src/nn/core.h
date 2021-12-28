#ifndef AZAH_NN_CORE_H_
#define AZAH_NN_CORE_H_

#include <vector>

#include "Eigen/Dense"

namespace azah {
namespace nn {

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Matrix<float, 1, Eigen::Dynamic> RowVector;
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> ColVector;

typedef std::vector<Matrix> BatchMatrix;
typedef std::vector<RowVector> BatchRowVector;
typedef std::vector<ColVector> BatchColVector;

inline BatchMatrix CreateBatchMatrix(int batch_n, int rows_n, int cols_n, 
																		 float initial_value = 0.0) {
	BatchMatrix batch_matrix(batch_n, Matrix(rows_n, cols_n));
	for (auto& matrix : batch_matrix) {
		matrix = Matrix::Constant(initial_value);
	}
	return batch_matrix;
}

inline BatchRowVector CreateBatchRowVector(int batch_n, int size_n,
																			     float initial_value = 0.0) {
	BatchRowVector batch_row_vector(batch_n, RowVector(1, size_n));
	for (auto& row_vector : batch_row_vector) {
		row_vector = RowVector::Constant(initial_value);
	}
	return batch_row_vector;
}

inline BatchColVector CreateBatchColVector(int batch_n, int size_n,
																					 float initial_value = 0.0) {
	BatchColVector batch_col_vector(batch_n, ColVector(1, size_n));
	for (auto& col_vector : batch_col_vector) {
		col_vector = ColVector::Constant(initial_value);
	}
	return batch_col_vector;
}

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_CORE_H_
