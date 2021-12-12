#include <math.h>
#include <stdint.h>

#include <chrono>
#include <iostream>
#include <vector>


inline float direct_swish(float x) {
	return x / (1.f + std::expf(x));
}

struct LinearSegment {
	LinearSegment(float m, float b) : m(m), b(b) {}

	const float m;
	const float b;
};

inline float lut_swish(float x, const std::vector<LinearSegment>& lut) {
	int i = static_cast<int>(x * 0.0625f) + 128;
	i = i > 255 ? 255 : (i < 0 ? 0 : i);
	const LinearSegment& segment = lut[i];
	return segment.m * x + segment.b;
}

const int64_t kDatasetLength = 1024;
const int64_t kRepeats = 16777216;
const int kRepeatMask = 0x3ff;

int main(int argc, char* argv[]) {
	std::vector<LinearSegment> segs;
	for (int i = 0; i < 256; ++i) {
		segs.emplace_back(i, 1.0 / (1.0 + i));
	}

	float sine_garbage[kDatasetLength];
	for (int i = 0; i < kDatasetLength; ++i) {
		sine_garbage[i] = std::sinf(static_cast<float>(i));
	}

	float acc = 0;

	auto t1 = std::chrono::high_resolution_clock::now();
	for (int64_t i = 0; i < kDatasetLength * kRepeats; ++i) {
		acc += lut_swish(sine_garbage[i & kRepeatMask], segs);
		//acc += direct_swish(sine_garbage[i & kRepeatMask]);
	}
	auto t2 = std::chrono::high_resolution_clock::now();
	auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);  

	std::cout << acc << std::endl;
	std::cout << "Time: " << static_cast<double>(ms_int.count())
		/ (kRepeats * kDatasetLength);

	// lut : 1.13930e-06 + 1.13231e-06
	// fp  : 4.90109e-06 + 4.90586e-06

	// lut : 1.13
	// fp  : 4.90
	// ~4.34x faster

	return 0;
}
