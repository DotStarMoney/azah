#include "nn/graph.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "nn/builder.h"

namespace azah {
namespace nn {

struct Graph::Op {
  virtual void RunParallel(bool batch_mode, int batch_i) = 0;

  virtual void RunSynchronized() = 0;

  // Edges.
  const std::vector<SlotId> in;
  const std::vector<SlotId> out;

  const bool batch_synchronized;
};

namespace {



}  // namespace

absl::StatusOr<std::unique_ptr<Graph>> Graph::Build(const Builder& builder) {

}

Graph::Graph(std::string* consumed_description) : 
    description_(std::move(*consumed_description)) {}

// TODO: "GetOrCreateOpPlan"

/*
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
*/

}  // namespace nn
}  // namespace azah
