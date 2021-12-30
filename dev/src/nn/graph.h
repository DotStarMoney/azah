#ifndef AZAH_NN_GRAPH_H_
#define AZAH_NN_GRAPH_H_

#include <stddef.h>

#include <string>
#include <string_view>
#include <vector>

#include "nn/core.h"
#include "nn/builder.h"
#include "util/noncopyable.h"

namespace azah {
namespace nn {

/*
class Graph : public util::NonCopyable {
 public:
  typedef std::size_t SlotId;

  Graph(const Builder& builder);

  std::string_view description() const { return description_; }

  ColVector& input(SlotId input_id);
  ColVector& input(std::string_view input_name);
  BatchColVector& input_batch(SlotId input_id);
  BatchColVector& input_batch(std::string_view input_name);

  ColVector& output(SlotId output_id) const;
  ColVector& output(std::string_view output_name) const;

  SlotId input_name_to_id(std::string_view input_name) const;
  SlotId output_name_to_id(std::string_view output_name) const;

  void Forward(const std::vector<std::string_view>& output_names);
  void Forward(const std::vector<SlotId>& output_ids);

  void ForwardBatch();

  void BackwardBatch(SlotId output_id);
  void BackwardBatch(std::string_view output_name);

  std::size_t matrix_var_n() const;
  std::size_t vector_var_n() const;

  Matrix& matrix_var(SlotId id);
  ColVector& vector_var(SlotId id);
  Matrix& matrix_var_grad(SlotId id);
  ColVector& vector_var_grad(SlotId id);

  const std::vector<Matrix>& get_all_matrix_vars() const;
  const std::vector<ColVector>& get_all_vector_vars() const;

  // These consume their parameters.
  void set_all_matrix_vars(std::vector<Matrix>* consumed_vars);
  void set_all_vector_vars(std::vector<ColVector>* consumed_vars);

 private:
  const std::string description_;

  // Includes inputs and outputs.
  std::vector<BatchColVector> intermediates_;
  std::vector<SlotId> intermediate_to_grad_;
  std::vector<BatchColVector> intermediate_grads_;

  std::vector<Matrix> matrix_vars_;
  std::vector<Matrix> matrix_var_grads_;
  // Matrix variables available to clients for gradient calculations.
  std::vector<SlotId> external_matrix_vars_;

  std::vector<ColVector> vector_vars_;
  std::vector<ColVector> vector_var_grads_;
  // Vector variables available to clients for gradient calculations.
  std::vector<SlotId> external_vector_vars_;
};
*/

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_BUILDER_H_
