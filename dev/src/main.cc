#include <iostream>

#include "nn/constant.h"
#include "nn/data_types.h"
#include "nn/op/add.h"
#include "nn/op/fork.h"
#include "nn/op/group_matmul.h"
#include "nn/op/matmul.h"
#include "nn/op/multiply.h"
#include "nn/op/swish.h"
#include "nn/op/tanh.h"
#include "nn/variable.h"

#include "nn/op/group.h"
#include "nn/op/ungroup.h"

// Next up...
// 
// Softmax

using azah::nn::Matrix;

int main(int argc, char* argv[]) {
  /*
  Matrix<2, 2> a;
  a << 0.5, 0.6, 0.7, 0.8;

  Matrix<2, 1> b;
  b << 0.1, 0.2;

  auto x = azah::nn::Variable<2, 2>(a);
  auto y = azah::nn::Variable<2, 1>(b);
  
  auto xy = azah::nn::op::GroupMatmul<2, 2, 2, 2, 1>(x, y);

  auto forked_xy = azah::nn::op::Fork(xy);

  auto tanned = azah::nn::op::TanH(forked_xy);

  Matrix<4, 1> c;
  c << 3, 6, 9, 12;
  auto z = azah::nn::Constant<4, 1>(c);
  auto mulled = azah::nn::op::Multiply(forked_xy, z);

  auto sum = azah::nn::op::Add(tanned, mulled);
  auto final_swish = azah::nn::op::Swish(sum);

  std::cout << "result=\n" << final_swish.output(0) << "\n";
  
  final_swish.backprop(0);

  std::cout << "gradient x=\n" << x.gradient() << "\n";
  std::cout << "gradient y=\n" << y.gradient() << "\n";
  */
  Matrix<4, 3> v;
  v << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
  auto s = azah::nn::Variable<4, 3>(v);

  auto s_grouped = azah::nn::op::Group<2, 4, 3>(s);
  auto s_ungrouped = azah::nn::op::Ungroup<2, 4, 3>(s_grouped);

  std::cout << "result=\n" << s_ungrouped.output(0) << "\n";

  s_ungrouped.backprop(0);

  std::cout << "gradient x=\n" << s.gradient() << "\n";

  return 0;
}
