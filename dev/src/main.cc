#include <iostream>

#include "nn/core.h"

int main(int argc, char* argv[]) {

	auto init = azah::nn::CreateUniformRandomInitializer(0, 1);

	auto m1 = azah::nn::CreateBatchMatrix(4, 1024, 256, init);
	auto m2 = azah::nn::CreateBatchColVector(4, 256, init);

	auto result = m1[0] * m2[0];

	std::cout << result.rows() << ", " << result.cols();

	return 0;
}
