#include <iostream>

#include "nn/constant.h"
#include "nn/data_types.h"
#include "nn/op/add.h"
#include "nn/op/fork.h"
#include "nn/op/scalar_fmadd.h"
#include "nn/op/scalar_mse.h"
#include "nn/op/group_matmul.h"
#include "nn/op/matmul.h"
#include "nn/op/mean.h"
#include "nn/op/multiply.h"
#include "nn/op/swish.h"
#include "nn/op/tanh.h"
#include "nn/variable.h"

// Next up...
// 
// Softmax

using azah::nn::Matrix;

int main(int argc, char* argv[]) {

  Matrix<2, 2> xx;
  xx << 1.0, 3.0, 5.0, 3.0;

  Matrix<1, 1> mm;
  mm << 2.0;

  Matrix<1, 1> vv;
  vv << 2.0;

  auto x = azah::nn::Variable<2, 2>(xx);
  auto mean = azah::nn::Variable<1, 1>(mm);
  auto var = azah::nn::Variable<1, 1>(vv);


  auto fmadd = azah::nn::op::ScalarMSE(x, t);

  std::cout << "result=\n" << fmadd.output(0) << "\n";
  
  fmadd.backprop(0);

  std::cout << "gradient x=\n" << x.gradient() << "\n";
  std::cout << "gradient t=\n" << t.gradient() << "\n";

  return 0;
}
