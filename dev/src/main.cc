#include <iostream>

#include "nn/constant.h"
#include "nn/init.h"
#include "nn/network.h"
#include "nn/op/layer_norm.h"
#include "nn/op/matmul.h"
#include "nn/op/mse.h"
#include "nn/op/swish.h"
#include "nn/variable.h"

namespace {

class SpirolNet : public azah::nn::Network {
 public:
   SpirolNet() :
       input_(azah::nn::init::Zeros<2, 1>()),
       dense1_k_(azah::nn::init::GlorotUniform<32, 2>()),
       dense1_(dense1_k_, input_),
       norm1_b_(azah::nn::init::Zeros<1, 1>()),
       norm1_g_(azah::nn::init::Ones<1, 1>()),
       norm1_(dense1_, norm1_b_, norm1_g_),
       swish1_(norm1_),
       dense2_k_(azah::nn::init::GlorotUniform<32, 32>()),
       dense2_(dense2_k_, swish1_),
       norm2_b_(azah::nn::init::Zeros<1, 1>()),
       norm2_g_(azah::nn::init::Ones<1, 1>()),
       norm2_(dense2_, norm2_b_, norm2_g_),
       swish2_(norm2_),
       linear_k_(azah::nn::init::GlorotUniform<3, 32>()),
       linear_(linear_k_, swish2_),
       target_(azah::nn::init::Zeros<3, 1>()),
       loss_(linear_, target_) {
     AddOutput(&linear_);
     AddTarget(&loss_);

     AddVariable(&dense1_k_);
     AddVariable(&norm1_b_);
     AddVariable(&norm1_g_);
     AddVariable(&dense2_k_);
     AddVariable(&norm2_b_);
     AddVariable(&norm2_g_);
     AddVariable(&linear_k_);

     AddConstant(&input_);
     AddConstant(&target_);
   }

 private:
  azah::nn::Constant<2, 1> input_;

  azah::nn::Variable<32, 2> dense1_k_;
  azah::nn::op::Matmul<32, 2, 2, 1> dense1_;

  azah::nn::Variable<1, 1> norm1_b_;
  azah::nn::Variable<1, 1> norm1_g_;
  azah::nn::op::LayerNorm<32, 1> norm1_;

  azah::nn::op::Swish<32, 1> swish1_;

  azah::nn::Variable<32, 32> dense2_k_;
  azah::nn::op::Matmul<32, 32, 32, 1> dense2_;

  azah::nn::Variable<1, 1> norm2_b_;
  azah::nn::Variable<1, 1> norm2_g_;
  azah::nn::op::LayerNorm<32, 1> norm2_;

  azah::nn::op::Swish<32, 1> swish2_;

  azah::nn::Variable<3, 32> linear_k_;
  azah::nn::op::Matmul<3, 32, 32, 1> linear_;

  azah::nn::Constant<3, 1> target_;
  azah::nn::op::MSE<3, 1> loss_;
};

}  // namespace;

// TODO:
//   Network sub-classes should do the initialization of variables.

int main(int argc, char* argv[]) {
  SpirolNet model;
  /*
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
  */
  return 0;
}
