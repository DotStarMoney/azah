#ifndef AZAH_IO_SERIALIZABLE_H_
#define AZAH_IO_SERIALIZABLE_H_

#include <iostream>

namespace azah {
namespace io {

class Serializable {
 public:
  virtual void Serialize(std::ostream& out) const = 0;
  virtual void Deserialize(std::istream& in) = 0;
};

}  // namespace io
}  // namespace azah

#endif  // AZAH_IO_SERIALIZABLE_H_
