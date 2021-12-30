#include <stdint.h>

#include <chrono>
#include <functional>
#include <iostream>
#include <thread>

#include "async/work_queue.h"
#include "util/random.h"

void work_fn(uint32_t worker_index, azah::async::WorkQueue* queue, 
						 int total[]) {
	total[worker_index]++;
	std::this_thread::sleep_for(std::chrono::milliseconds(
		  1 + azah::util::rnd() % 10));
	auto fn = std::bind(work_fn, std::placeholders::_1, queue, total);
	if (azah::util::TrueWithChance(0.5)) queue->AddWork(fn);
}

int main(int argc, char* argv[]) {
	azah::async::WorkQueue queue(7, 1024);

	int total[7] = { 0, 0, 0, 0, 0, 0, 0};

	auto fn = std::bind(work_fn, std::placeholders::_1, &queue, total);

	std::cout << "Letting threads start (5s)...\n";
	std::this_thread::sleep_for(std::chrono::milliseconds(5000));

	for (int i = 0; i < 256; ++i) {
		queue.AddWork(fn);
	}
	
	queue.Join();

	for (int i = 0; i < 7; ++i) {
		std::cout << total[i] << ", ";
	}
	std::cout << '\n';

	return 0;
}
