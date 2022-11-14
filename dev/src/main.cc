#include <iostream>

#include "nn/constant.h"
#include "nn/data_types.h"
#include "nn/op/add.h"
#include "nn/op/concat.h"
#include "nn/op/fork.h"
#include "nn/op/scalar_fmadd.h"
#include "nn/op/scalar_mse.h"
#include "nn/op/group_matmul.h"
#include "nn/op/layer_norm.h"
#include "nn/op/matmul.h"
#include "nn/op/softmax_cross_ent.h"
#include "nn/op/mean.h"
#include "nn/op/multiply.h"
#include "nn/op/swish.h"
#include "nn/op/tanh.h"
#include "nn/op/z_score.h"
#include "nn/op/transpose.h"
#include "nn/variable.h"

// TODO:
//   Add subtract + square ops to speed up zscore (only subtract mean once)
//   Network sub-classes should do the initialization of variables.

int main(int argc, char* argv[]) {
  azah::nn::Matrix<2, 2> y_true_m;
  y_true_m << 0.05, 0.2, 0.6, 0.15;
  
  azah::nn::Matrix<2, 2> y_pred_m;
  y_pred_m << 30.0, 30.0, 100.0, 10.0;

  auto y_true = azah::nn::Constant<2, 2>(y_true_m);
  auto y_pred = azah::nn::Variable<2, 2>(y_pred_m);

  auto y_pred_t = azah::nn::op::Transpose(y_pred);
  auto cat = azah::nn::op::Multiply(y_pred_t, y_true);

  std::cout << "result=\n" << cat.OutputBase(0) << "\n";
  
  cat.BackpropBase(0);

  std::cout << "gradient y_pred=\n" << y_pred.gradient_base() << "\n";

  return 0;
}
