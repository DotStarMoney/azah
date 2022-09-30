#include <iostream>

#include "nn/constant.h"
#include "nn/data_types.h"
#include "nn/op/add.h"
#include "nn/op/multiply.h"
#include "nn/op/swish.h"
#include "nn/variable.h"

using azah::nn::Matrix;

int main(int argc, char* argv[]) {
  // Prepare the column vector [[0.1], [0.2], [0.3], [0.4]]
  Matrix<4, 1> s;
  s << 0.1,
       0.2,
       0.3,
       0.4;

  // Evaluate z = f(2x + f(x))

  Matrix<4, 1> two_m;
  two_m << 2,
           2,
           2,
           2;
  auto two = azah::nn::Constant<4, 1>(two_m);
  auto x = azah::nn::Variable<4, 1>(s);

  auto fx = azah::nn::op::Swish(x);
  auto two_x = azah::nn::op::Multiply(two, x);
  auto xpfx = azah::nn::op::Add(two_x, fx);
  auto z = azah::nn::op::Swish(xpfx);

  std::cout << "result=\n" << z.output(0) << "\n";

  // Find the derivative of z with respect to x
  Matrix<4, 1> ones = Matrix<4, 1>::Constant(1);
  z.backprop(0, ones);

  std::cout << "gradient=\n" << x.gradient() << "\n";

  return 0;
}
