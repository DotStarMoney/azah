#ifndef AZAH_UTIL_NONCOPYABLE_H_
#define AZAH_UTIL_NONCOPYABLE_H_

// Utility class that, when inherited, disables copy and assignment for the
// inheritor.
//
// Use it like this:
//
// class PleaseDontCopyMe : public NonCopyable {
//   ...
// }
//
namespace azah {
namespace util {

class NonCopyable {
 public:
  NonCopyable() = default;

  NonCopyable(const NonCopyable&) = delete;
  NonCopyable& operator=(const NonCopyable&) = delete;
};

}  // namespace util
}  // namespace azah

#endif  // AZAH_UTIL_NONCOPYABLE_H_
