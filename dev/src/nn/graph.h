#ifndef AZAH_NN_GRAPH_H_
#define AZAH_NN_GRAPH_H_

#include <stddef.h>

#include <string>
#include <string_view>
#include <vector>

#include "core.h"
#include "builder.h"

namespace azah {
namespace nn {

class Graph {
 public:
  typedef std::size_t SlotId;

  Graph(const Builder& builder);

  std::string_view description() const { return description_; }

  ColVector& input(SlotId id);
  ColVector& input(std::string_view name);
  BatchColVector& input_batch(SlotId id);
  BatchColVector& input_batch(std::string_view name);

  ColVector& output(SlotId id) const;
  ColVector& output(std::string_view name) const;

  SlotId input_name_to_id(std::string_view name) const;
  SlotId output_name_to_id(std::string_view name) const;

  void Forward();
  void ForwardBatch();

  //void BackwardBatch(/* some objective thing */);

  std::size_t matrix_var_n() const;
  std::size_t vector_var_n() const;

  Matrix& matrix_var(SlotId id);
  ColVector& vector_var(SlotId id);

  const std::vector<Matrix>& get_all_matrix_vars() const;
  const std::vector<ColVector>& get_all_vector_vars() const;

  // These consume their parameters.
  void set_all_matrix_vars(std::vector<Matrix>* vars);
  void set_all_vector_vars(std::vector<ColVector>* vars);

  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;

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

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_BUILDER_H_
