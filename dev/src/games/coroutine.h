#ifndef AZAH_GAMES_COROUTINE_H_
#define AZAH_GAMES_COROUTINE_H_

#include <coroutine>

namespace azah {
namespace games {
namespace coroutine {

struct Void {
  struct promise_type {
    Void get_return_object() {
      return {
        .handle = std::coroutine_handle<promise_type>::from_promise(*this)
      };
    }
    std::suspend_never initial_suspend() { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void unhandled_exception() {}
    void return_void() {}
  };
  std::coroutine_handle<promise_type> handle;
  operator std::coroutine_handle<promise_type>() const { return handle; }
  operator std::coroutine_handle<>() const { return handle; }
};

using VoidHandle = std::coroutine_handle<Void::promise_type>;

}  // namespace coroutine
}  // namespace games
}  // namespace azah

#endif AZAH_GAMES_COROUTINE_H_
