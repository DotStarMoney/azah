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
#include "nn/op/transpose.h"
#include "nn/op/row_mean.h"
#include "nn/op/broadcast_matmul.h"
#include "nn/variable.h"

// TODO:
//   Network sub-classes should do the initialization of variables.

int main(int argc, char* argv[]) {
  azah::nn::Matrix<4, 2> y_true_m;
  y_true_m << 0.05, 0.2, 0.6, 0.15, 10.0, 9.0, 11.0, 12.0;
  
  azah::nn::Matrix<2, 1> y_pred_m;
  y_pred_m << 0.3, 0.1;

  auto y_true = azah::nn::Constant<4, 2>(y_true_m);
  auto y_pred = azah::nn::Variable<2, 1>(y_pred_m);

  auto bm = azah::nn::op::BroadcastMatmul<4, 2, 2>(y_true, y_pred);
  auto s = azah::nn::op::Swish(bm);
  auto mean = azah::nn::op::Mean(s);

  std::cout << "result=\n" << mean.OutputBase(0) << "\n";
  
  mean.BackpropBase(0);

  std::cout << "gradient y_pred=\n" << y_pred.gradient_base() << "\n";

  return 0;
}
