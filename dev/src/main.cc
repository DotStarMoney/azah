#include "nn/builder.h"
#include "nn/graph.h"

int main(int argc, char* argv[]) {
	azah::nn::Builder graph_builder;

	auto x = graph_builder.Input("board_player_1", 24);
	x = graph_builder.Dense(x, 64);
	x = graph_builder.BatchNormalization(x);
	x = graph_builder.Swish(x);
	graph_builder.Output(x, "move_prob");

	azah::nn::Graph network(graph_builder);

	auto input_vec = network.input("board_player_1");
	// input_vec = some input state

	network.Forward();

	auto output = network.output("move_prob");

	auto input_batch = network.input_batch("board_player_1");
	network.ForwardBatch();
	//network.BackwardBatch(ObjectFnObject);

	return 0;
}
