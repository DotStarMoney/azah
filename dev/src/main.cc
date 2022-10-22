#include <iostream>

#include "nn/constant.h"
#include "nn/data_types.h"
#include "nn/op/add.h"
#include "nn/op/fork.h"
#include "nn/op/scalar_fmadd.h"
#include "nn/op/scalar_mse.h"
#include "nn/op/group_matmul.h"
#include "nn/op/layer_norm.h"
#include "nn/op/matmul.h"
#include "nn/op/mean.h"
#include "nn/op/multiply.h"
#include "nn/op/swish.h"
#include "nn/op/tanh.h"
#include "nn/op/z_score.h"
#include "nn/variable.h"

// Next up...
// 
// Softmax

using azah::nn::Matrix;

int main(int argc, char* argv[]) {
  Matrix<2, 2> xx;
  xx << 3.0, 9.0, 5.0, 10.0;

  Matrix<1, 1> mm;
  mm << 0.0;

  Matrix<1, 1> vv;
  vv << 1.0;

  auto x = azah::nn::Variable<2, 2>(xx);
  auto beta = azah::nn::Variable<1, 1>(mm);
  auto gamma = azah::nn::Variable<1, 1>(vv);

  auto layer_norm = azah::nn::op::LayerNorm(x, beta, gamma);
  auto z = azah::nn::op::Mean(layer_norm);

  std::cout << "result=\n" << z.output(0) << "\n";
  
  z.backprop(0);

  std::cout << "gradient x=\n" << x.gradient() << "\n";
  std::cout << "gradient beta=\n" << beta.gradient() << "\n";
  std::cout << "gradient gamma=\n" << gamma.gradient() << "\n";

  return 0;
}
