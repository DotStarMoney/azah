#include "async/work_queue.h"

#include <chrono>
#include <functional>
#include <thread>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "util/random.h"

namespace azah {
namespace async {
namespace {

void work_fn(uint32_t worker_index, azah::async::WorkQueue* queue,
	int total[]) {
	total[worker_index]++;
	std::this_thread::sleep_for(std::chrono::milliseconds(
		1 + azah::util::rnd() % 10));
	auto fn = std::bind(work_fn, std::placeholders::_1, queue, total);
	if (azah::util::TrueWithChance(0.5)) queue->AddWork(fn);
}

}  // namespace

TEST(WorkQueueTest, MultiProducerMultiConsumer) {
	azah::async::WorkQueue queue(7, 1024);

	int total[7] = { 0, 0, 0, 0, 0, 0, 0 };

	auto fn = std::bind(work_fn, std::placeholders::_1, &queue, total);

	std::this_thread::sleep_for(std::chrono::milliseconds(1000));

	for (int i = 0; i < 256; ++i) {
		queue.AddWork(fn);
	}
	queue.Join();

	EXPECT_THAT(total, testing::Each(testing::Gt(2)));
}

}  // namespace async
}  // namespace azah
