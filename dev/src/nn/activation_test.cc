#include "activation.h"

#include <math.h>

#include "gtest/gtest.h"

namespace azah {
namespace nn {
namespace {
   
inline float swish(float x) {
  return x / (1.f + std::expf(-x));
}

inline float swish_d(float x) {
  float s = swish(x);
  return s + (1.f - s) / (1.f + std::expf(-x));
}

inline float sigmoid(float x) {
  return 1.f / (1.f + std::expf(-x));
}

inline float sigmoid_d(float x) {
  float s = sigmoid(x);
  return s * (1 - s);
}

inline float tanh(float x) {
  return std::tanhf(x);
}

inline float tanh_d(float x) {
  float t = tanh(x);
  return 1 - t * t;
}

}  // namespace

TEST(ActivationTest, FastSwish) {
  for (float x = -8.f; x < -2.f; x += 0.1f) {
    EXPECT_NEAR(swish(x), FastSwish(x), 0.005f);
  }

  // We're more precise towards zero.
  for (float x = -2.f; x < 2.f; x += 0.1f) {
    EXPECT_NEAR(swish(x), FastSwish(x), 0.001f);
  }

  for (float x = 2.f; x < 8.f; x += 0.1f) {
    EXPECT_NEAR(swish(x), FastSwish(x), 0.005f);
  }

  EXPECT_EQ(FastSwish(-10), 0.f);
  EXPECT_EQ(FastSwish(10), 10.f);
}

TEST(ActivationTest, FastSwishD) {
  for (float x = -8.f; x < -2.f; x += 0.1f) {
    EXPECT_NEAR(swish_d(x), FastSwishD(x), 0.005f);
  }

  // We're more precise towards zero.
  for (float x = -2.f; x < 2.f; x += 0.1f) {
    EXPECT_NEAR(swish_d(x), FastSwishD(x), 0.001f);
  }

  for (float x = 2.f; x < 8.f; x += 0.1f) {
    EXPECT_NEAR(swish_d(x), FastSwishD(x), 0.005f);
  }

  EXPECT_EQ(FastSwishD(-10), 0.f);
  EXPECT_EQ(FastSwishD(10), 1.f);
}

TEST(ActivationTest, FastSigmoid) {
  for (float x = -8.f; x < -2.f; x += 0.1f) {
    EXPECT_NEAR(sigmoid(x), FastSigmoid(x), 0.005f);
  }

  // We're more precise towards zero.
  for (float x = -2.f; x < 2.f; x += 0.1f) {
    EXPECT_NEAR(sigmoid(x), FastSigmoid(x), 0.001f);
  }

  for (float x = 2.f; x < 8.f; x += 0.1f) {
    EXPECT_NEAR(sigmoid(x), FastSigmoid(x), 0.005f);
  }

  EXPECT_EQ(FastSigmoid(-10), 0.f);
  EXPECT_EQ(FastSigmoid(10), 1.f);
}

TEST(ActivationTest, FastSigmoidD) {
  for (float x = -8.f; x < -2.f; x += 0.1f) {
    EXPECT_NEAR(sigmoid_d(x), FastSigmoidD(x), 0.005f);
  }

  // We're more precise towards zero.
  for (float x = -2.f; x < 2.f; x += 0.1f) {
    EXPECT_NEAR(sigmoid_d(x), FastSigmoidD(x), 0.001f);
  }

  for (float x = 2.f; x < 8.f; x += 0.1f) {
    EXPECT_NEAR(sigmoid_d(x), FastSigmoidD(x), 0.005f);
  }

  EXPECT_EQ(FastSigmoidD(-10), 0.f);
  EXPECT_EQ(FastSigmoidD(10), 0.f);
}

TEST(ActivationTest, FastTanh) {
  for (float x = -8.f; x < -2.f; x += 0.1f) {
    EXPECT_NEAR(tanh(x), FastTanh(x), 0.005f);
  }

  // We're more precise towards zero.
  for (float x = -2.f; x < 2.f; x += 0.1f) {
    EXPECT_NEAR(tanh(x), FastTanh(x), 0.001f);
  }

  for (float x = 2.f; x < 8.f; x += 0.1f) {
    EXPECT_NEAR(tanh(x), FastTanh(x), 0.005f);
  }

  EXPECT_EQ(FastTanh(-10), -1.f);
  EXPECT_EQ(FastTanh(10), 1.f);
}

TEST(ActivationTest, FastTanhD) {
  for (float x = -8.f; x < -2.f; x += 0.1f) {
    EXPECT_NEAR(tanh_d(x), FastTanhD(x), 0.005f);
  }

  // We're more precise towards zero, but not by much since the peak is sharper
  // than our linear approximation (double wide at 0) can handle.
  for (float x = -2.f; x < 2.f; x += 0.1f) {
    EXPECT_NEAR(tanh_d(x), FastTanhD(x), 0.003f);
  }

  for (float x = 2.f; x < 8.f; x += 0.1f) {
    EXPECT_NEAR(tanh_d(x), FastTanhD(x), 0.005f);
  }

  EXPECT_EQ(FastTanhD(-10), 0.f);
  EXPECT_EQ(FastTanhD(10), 0.f);
}

}  // namespace nn
}  // namespace azah
