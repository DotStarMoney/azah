#include <iostream>

#include "nn/constant.h"
#include "nn/data_types.h"
#include "nn/op/add.h"
#include "nn/op/group_matmul.h"
#include "nn/op/matmul.h"
#include "nn/op/multiply.h"
#include "nn/op/swish.h"
#include "nn/variable.h"

// Next up...
// 
// Softmax

using azah::nn::Matrix;

int main(int argc, char* argv[]) {
  Matrix<2, 2> a;
  a << 5, 6, 7, 8;

  Matrix<2, 1> b;
  b << 1, 2;

  auto x = azah::nn::Variable<2, 2>(a);
  auto y = azah::nn::Variable<2, 1>(b);
  
  auto xy = azah::nn::op::GroupMatmul<2, 2, 2, 2, 1>(x, y);

  std::cout << "result=\n" << xy.output(0) << "\n";
  
  xy.backprop(0);

  std::cout << "gradient x=\n" << x.gradient() << "\n";
  std::cout << "gradient y=\n" << y.gradient() << "\n";

  return 0;
}
