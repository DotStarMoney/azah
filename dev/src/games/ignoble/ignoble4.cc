#include "ignoble4.h"

namespace azah {
namespace games {
namespace ignoble {

Ignoble4::Ignoble4() :
    round_phase_(RoundPhase::kTeamSelect),
    soil_n_{0, 0, 0, 0},
    herb_n_{0, 0, 0, 0},
    beast_n_{0, 0, 0, 0},
    coin_n_{0, 0, 0, 0} {}


}  // namespace ignoble
}  // namespace games
}  // namespace azah
