#include "core/half.h"
#include "core/rng.h"

#include <bit>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <limits>

// Référence : la conversion du compilateur vers _Float16, le type demi-précision de g++ et clang
namespace {
std::uint16_t reference(float value) {
   return std::bit_cast<std::uint16_t>(static_cast<_Float16>(value));
}
} // namespace

TEST_CASE("demi-précision : valeurs remarquables", "[half]") {
   CHECK(rt::float_to_half_bits(0.0f) == 0x0000);
   CHECK(rt::float_to_half_bits(-0.0f) == 0x8000);
   CHECK(rt::float_to_half_bits(1.0f) == 0x3C00);
   CHECK(rt::float_to_half_bits(-2.0f) == 0xC000);
   CHECK(rt::float_to_half_bits(65504.0f) == 0x7BFF); // plus grande valeur finie
   CHECK(rt::float_to_half_bits(1e6f) == 0x7C00);     // trop grand : infini
   CHECK(rt::float_to_half_bits(std::numeric_limits<float>::infinity()) == 0x7C00);
   CHECK((rt::float_to_half_bits(std::numeric_limits<float>::quiet_NaN()) & 0x7FFF) > 0x7C00);
   CHECK(rt::float_to_half_bits(5.9604645e-8f) == 0x0001); // plus petit sous-normal
}

TEST_CASE("demi-précision : identique au compilateur sur un million de valeurs", "[half]") {
   rt::Pcg32 rng(16, 0);
   int mismatches = 0;
   for (int i = 0; i < 1'000'000; ++i) {
      // Motifs de bits quelconques (normaux, sous-normaux, grands), sauf les NaN
      const auto value = std::bit_cast<float>(rng.next_u32());
      if (value != value) {
         continue;
      }
      if (rt::float_to_half_bits(value) != reference(value)) {
         ++mismatches;
         INFO("valeur " << value);
         CHECK(rt::float_to_half_bits(value) == reference(value));
      }
   }
   // Toutes les valeurs de la plage utile d'une image, du sous-normal à la saturation
   for (std::uint32_t half = 0; half < 0x7C00; ++half) {
      const auto value = static_cast<float>(
         std::bit_cast<_Float16>(static_cast<std::uint16_t>(half))
      );
      mismatches += rt::float_to_half_bits(value) != half ? 1 : 0;
   }
   CHECK(mismatches == 0);
}
