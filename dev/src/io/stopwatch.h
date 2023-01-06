#ifndef AZAH_IO_STOPWATCH_H_
#define AZAH_IO_STOPWATCH_H_

#include <stddef.h>

#include <chrono>

namespace azah {
namespace io {

// Not thread-safe.
class Stopwatch {
 public:
  Stopwatch(const Stopwatch&);
  Stopwatch(Stopwatch&&) noexcept;
  Stopwatch& operator=(const Stopwatch&);
  Stopwatch& operator=(Stopwatch&&) noexcept;

  Stopwatch();

  void Start();
  void End();
    
  double average_seconds() const;
  double total_seconds() const;
  std::size_t total_timings() const;
 
 private:
  void MetricsCheckReady() const;

  bool timing_;
  std::size_t timing_n_;
  double acc_seconds_;
  std::chrono::steady_clock::time_point start_time_;
};

}  // namespace mancala
}  // namespace io

#endif
