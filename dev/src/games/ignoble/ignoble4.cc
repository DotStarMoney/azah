#include "ignoble4.h"

#include <array>
#include <string_view>
#include <vector>

#include "../../nn/data_types.h"
#include "../game.h"
#include "absl/random/bit_gen_ref.h"

namespace azah {
namespace games {
namespace ignoble {

Ignoble4::Ignoble4() :
    soil_n_{0, 0, 0, 0},
    herb_n_{0, 0, 0, 0},
    beast_n_{0, 0, 0, 0},
    coin_n_{0, 0, 0, 0} {}

const std::string_view Ignoble4::name() const {
  return kName_;
}

int Ignoble4::CurrentPlayerI() const {
  return 0;
}

int Ignoble4::CurrentMovesN() const {
  return 0;
}

GameState Ignoble4::State() const {
  return GameState::kUnknown;
}

std::array<float, 4> Ignoble4::Outcome() const {
  return {0.0f, 0.0f, 0.0f, 0.0f};
}

std::vector<nn::DynamicMatrix> Ignoble4::StateToMatrix() const {
  return {};
}

int Ignoble4::PolicyClassI() const {
  return 0;
}

float Ignoble4::PolicyForMoveI(const nn::DynamicMatrix& policy,
                               int move_i) const {
  return 0.0f;
}

nn::DynamicMatrix Ignoble4::PolicyMask() const {
  return nn::DynamicMatrix();
}

void Ignoble4::MakeMove(int move_i, absl::BitGenRef bitgen) {
  //
}

}  // namespace ignoble
}  // namespace games
}  // namespace azah
