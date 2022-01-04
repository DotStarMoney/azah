#include <iostream>

#include "nn/builder.h"
#include "nn/graph.h"

int main(int argc, char* argv[]) {

	azah::nn::Builder b;

	auto s = b.Input("cool", 256);
	s = b.Multiply(s, 0.1f);
	s = b.Tanh(s);
	b.Output(s, "cool_output");

	

	return 0;
}
