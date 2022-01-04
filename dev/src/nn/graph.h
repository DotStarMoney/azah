#ifndef AZAH_NN_GRAPH_H_
#define AZAH_NN_GRAPH_H_

#include <stddef.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "nn/core.h"
#include "nn/builder.h"
#include "util/noncopyable.h"

namespace azah {
namespace nn {


class Graph : public util::NonCopyable {
 public:
  typedef std::size_t SlotId;

  static absl::StatusOr<std::unique_ptr<Graph>> Build(const Builder& builder);

  std::string_view description() const { return description_; }

  ColVector& input(SlotId input_id, int batch_i = 0);
  ColVector& input(std::string_view input_name, int batch_i = 0);
  BatchColVector& input_batch(SlotId input_id);
  BatchColVector& input_batch(std::string_view input_name);

  const ColVector& output(SlotId output_id, int batch_i = 0) const;
  const ColVector& output(std::string_view output_name, int batch_i = 0) const;
  const BatchColVector& output_batch(SlotId output_id) const;
  const BatchColVector& output_batch(std::string_view output_name) const;

  SlotId input_name_to_id(std::string_view input_name) const;
  SlotId output_name_to_id(std::string_view output_name) const;

  void Forward(const std::vector<std::string_view>& output_names, 
               int batch_i = 0);
  void Forward(const std::vector<SlotId>& output_ids, int batch_i = 0);
  void Forward(std::string_view output_name, int batch_i = 0);
  void Forward(SlotId output_id, int batch_i = 0);

  void ForwardBatch(const std::vector<std::string_view>& output_names);
  void ForwardBatch(const std::vector<SlotId>& output_ids);
  void ForwardBatch(std::string_view output_name);
  void ForwardBatch(SlotId output_id);

  void BackwardBatch(SlotId output_id);
  void BackwardBatch(std::string_view output_name);

  std::size_t matrix_var_n() const;
  std::size_t vector_var_n() const;

  Matrix& matrix_var(SlotId id);
  ColVector& vector_var(SlotId id);
  Matrix& matrix_var_grad(SlotId id);
  ColVector& vector_var_grad(SlotId id);

  const std::vector<Matrix>& get_all_matrix_data() const;
  const std::vector<ColVector>& get_all_vector_data() const;

  // These consume their parameters.
  void set_all_matrix_data(std::vector<Matrix>* consumed_data);
  void set_all_vector_data(std::vector<ColVector>* consumed_data);

 private:
  Graph(std::string* consumed_description);

  const std::string description_;

  // Includes inputs and outputs.
  std::vector<BatchColVector> intermediates_;
  std::vector<SlotId> intermediate_to_grad_;
  std::vector<BatchColVector> intermediate_grads_;

  std::vector<Matrix> matrix_vars_;
  std::vector<Matrix> matrix_var_grads_;
  // Matrix variables that have gradients.
  std::vector<SlotId> grad_path_matrix_vars_;

  std::vector<ColVector> vector_vars_;
  std::vector<ColVector> vector_var_grads_;
  // Vector variables that have gradients.
  std::vector<SlotId> grad_path_vector_vars_;

  // An operation node.
  struct Op;

  friend struct SynchronizedOp;

  std::vector<Op> ops_;
};

}  // namespace nn
}  // namespace azah

#endif  // AZAH_NN_BUILDER_H_
