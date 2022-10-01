#include <iostream>

#include "nn/constant.h"
#include "nn/data_types.h"
#include "nn/op/add.h"
#include "nn/op/matmul.h"
#include "nn/op/multiply.h"
#include "nn/op/swish.h"
#include "nn/variable.h"


// Next up...
// 
// GroupMatmul
// GroupNorm
// Softmax


using azah::nn::Matrix;

int main(int argc, char* argv[]) {
  Matrix<2, 2> a;
  a << 5, 6, 7, 8;

  Matrix<2, 2> b;
  b << 1, 2, 3, 4;

  Matrix<2, 2> c;
  c << -1, -2, -3, -4;

  auto x = azah::nn::Variable<2, 2>(a);
  auto y = azah::nn::Variable<2, 2>(b);
  auto q = azah::nn::Variable<2, 2>(c);

  auto xy = azah::nn::op::Matmul(x, y);
  auto xyq = azah::nn::op::Matmul(xy, q);

  std::cout << "result=\n" << xyq.output(0) << "\n";

  Matrix<2, 2> ones = Matrix<2, 2>::Constant(1);
  xyq.backprop(0, ones);

  std::cout << "gradient x=\n" << x.gradient() << "\n";
  std::cout << "gradient y=\n" << y.gradient() << "\n";
  std::cout << "gradient q=\n" << q.gradient() << "\n";

  return 0;
}
