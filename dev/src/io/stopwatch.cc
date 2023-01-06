#include "stopwatch.h"

#include <stddef.h>

#include <chrono>

#include "glog/logging.h"

namespace azah {
namespace io {

Stopwatch::Stopwatch() : timing_(false), timing_n_(0), acc_seconds_(0.0) {}

Stopwatch::Stopwatch(const Stopwatch& other) {
  if (timing_) LOG(FATAL) << "Cannot copy while timing.";
  *this = other;
}

Stopwatch::Stopwatch(Stopwatch&& other) noexcept {
  if (timing_) LOG(FATAL) << "Cannot move while timing.";
  *this = std::move(other);
}

Stopwatch& Stopwatch::operator=(const Stopwatch& other) {
  if (timing_) LOG(FATAL) << "Cannot assign while timing.";
  *this = other;
  return *this;
}

Stopwatch& Stopwatch::operator=(Stopwatch&& other) noexcept {
  if (timing_) LOG(FATAL) << "Cannot move assign while timing.";
  *this = std::move(other);
  return *this;
}

void Stopwatch::Start() {
  if (timing_) LOG(FATAL) << "Must end timing before starting again.";
  timing_ = true;
  start_time_ = std::chrono::steady_clock::now();
}

void Stopwatch::End() {
  auto end_time = std::chrono::steady_clock::now();
  if (!timing_) LOG(FATAL) << "Timing has not started.";
  timing_ = false;
  ++timing_n_;
  acc_seconds_ += static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          end_time - start_time_).count()) * 1e-9;
}

double Stopwatch::average_seconds() const {
  MetricsCheckReady();
  return acc_seconds_ / static_cast<double>(timing_n_);
}

double Stopwatch::total_seconds() const {
  MetricsCheckReady();
  return acc_seconds_;
}

std::size_t Stopwatch::total_timings() const {
  MetricsCheckReady();
  return timing_n_;
}

void Stopwatch::MetricsCheckReady() const {
  if (timing_) LOG(FATAL) << "Cannot access metrics while timing.";
  if (timing_n_ == 0) LOG(FATAL) << "Cannot access metrics before any timing.";
}

}  // namespace mancala
}  // namespace io
