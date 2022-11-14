#include "adam.h"

#include <math.h>
#include <stdint.h>

#include <vector>

#include "data_types.h"
#include "glog/logging.h"
#include "network.h"

namespace azah {
namespace nn {
namespace {
 
static constexpr float kEpsilon = 1e-5;

}  // namespace

Adam::Adam(const Network& src, float beta1, float beta2) 
    : beta1_(beta1), beta2_(beta2) {
  std::vector<ConstDynamicMatrixRef> vars;
  src.GetVariables({}, vars);
  for (auto& var : vars) {
    m1_.push_back(DynamicMatrix::Zero(var.rows(), var.cols()));
    m2_.push_back(DynamicMatrix::Zero(var.rows(), var.cols()));
    updates_.push_back(0);
  }
}

void Adam::Update(
    float lr,
    const std::vector<uint32_t>& variables_i,
    const std::vector<ConstDynamicMatrixRef>& grads,
    Network& dest) {
  if (variables_i.empty() || grads.empty()) {
    LOG(FATAL) << "\"variables_i\" and \"grads\" cannot be empty.";
  }
  if (variables_i.size() != grads.size()) {
    LOG(FATAL) << "\"variables_i\" and \"grads\" must be the same size.";
  }

  std::vector<DynamicMatrixRef> vars;
  dest.GetVariables(variables_i, vars);
  for (int i = 0; i < variables_i.size(); ++i) {
    auto& grad = grads[i];
    auto& var = vars[i];

    uint32_t var_index = variables_i[i];
    auto& m1 = m1_[var_index];
    auto& m2 = m2_[var_index];
    float updates = ++(updates_[var_index]);

    m1 = (m1 - grad) * beta1_ + grad;
	auto m1_debias = m1.array() / (1.0f - std::powf(beta1_, updates));

	auto grad_2 = grad.array().square().matrix();
    m2 = (m2 - grad_2) * beta2_ + grad_2;
    auto m2_debias = m2.array() / (1.0f - std::powf(beta2_, updates));
    
	var -= lr * (m1_debias / (m2_debias + kEpsilon).sqrt()).matrix();
  }
}

}  // namespace nn
}  // namespace azah
