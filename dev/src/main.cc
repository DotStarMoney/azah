#include <math.h>

#include <iostream>
#include <vector>

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

float divide_no_nan(float x, float y) {
  return (y == 0.0f) ? 0.0f : x / y;
}

azah::nn::Matrix<3, 1> quad_blend(float x, float y, float alpha,
                                  const azah::nn::Matrix<3, 1>& c1,
                                  const azah::nn::Matrix<3, 1>& c2,
                                  const azah::nn::Matrix<3, 1>& c3) {
  float angle = std::atan2f(y, x);
  if (angle < alpha) {
    float prop = divide_no_nan(angle, alpha) * 0.5f + 0.5f;
    return ((c2.array() - c3.array()) * prop + c3.array()).matrix();
  } else if (angle > (1.57079632679f - alpha)) {
    float prop = divide_no_nan(1.57079632679f - angle, alpha) * 0.5f + 0.5f;
    return ((c2.array() - c1.array()) * prop + c1.array()).matrix();
  } else {
    return c2;
  } 
}

azah::nn::Matrix<3, 1> quad_color(
    float x, 
    float y, 
    float alpha, 
    const std::vector<azah::nn::Matrix<3, 1>>& colors) {
  if ((x == 0.0) && (y == 0.0)) {
    azah::nn::Matrix<3, 1> mean;
    for (const auto& c : colors) {
      mean = (mean.array() + c.array()).matrix();
    }
    mean = mean / static_cast<float>(colors.size());
    return mean;
  }

  azah::nn::Matrix<3, 1> c1;
  azah::nn::Matrix<3, 1> c2;
  azah::nn::Matrix<3, 1> c3;
  if (x < 0) {
    if (y < 0) {
      c1 = colors[1];
      c2 = colors[3];
      c3 = colors[2];
    } else {
      c1 = colors[0];
      c2 = colors[2];
      c3 = colors[3];
    }
  } else {
    if (y < 0) {
      c1 = colors[3];
      c2 = colors[1];
      c3 = colors[0];
    } else {
      c1 = colors[2];
      c2 = colors[0];
      c3 = colors[1];
    }
  }

  return quad_blend(std::abs(x), std::abs(y), alpha, c1, c2, c3);
}

unsigned char to_byte(float x) {
  return static_cast<unsigned char>(
      std::fmax(std::fmin(x * 255.0f, 255.0f), 0.0f));
}

void draw_spirol(int w, int h, float alpha, std::vector<float>& dest) {
  azah::nn::Matrix<3, 1> c1;
  c1 << 1.0f, 0.25f, 0.0f;
  azah::nn::Matrix<3, 1> c2;
  c2 << 0.25f, 0.75f, 0.0f;
  azah::nn::Matrix<3, 1> c3;
  c3 << 0.05f, 0.4f, 1.0f;
  azah::nn::Matrix<3, 1> c4;
  c4 << 1.0f, 0.75f, 0.25f;

  int offset = 0;
  for (int y = 0; y < h; ++y) {
    float yp = static_cast<float>(y) / h * 2.0f - 1.0f;
    for (int x = 0; x < w; ++x) {
      float xp = static_cast<float>(x) / w * 2.0f - 1.0f;

      float t = std::atan2f(yp, xp);
      float r = std::sqrtf(xp * xp + yp * yp);
      float tn = t + std::sqrtf(r) * 8.0f;
      
      float xn = std::cosf(tn) * r;
      float yn = std::sinf(tn) * r;

      auto color = quad_color(xn, yn, alpha, {c1, c2, c3, c4});

      dest[offset + 0] = color.coeff(2);
      dest[offset + 1] = color.coeff(1);
      dest[offset + 2] = color.coeff(0);

      offset += 3;
    }
  }
}

}  // namespace;

int main(int argc, char* argv[]) {
  std::vector<float> src(512 * 512 * 3, 0.0f);
  draw_spirol(512, 512, 0.1f, src);

  //SpirolNet model;


  return 0;
}
