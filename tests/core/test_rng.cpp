#include "core/rng.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdint>

namespace {
// Sorties du programme de démonstration officiel de PCG32 pour la graine 42 et le flux 54
constexpr std::array<std::uint32_t, 6> kReferenceOutputs{
   0xa15c02b7u, 0x7b47f409u, 0xba1d3330u, 0x83d2f293u, 0xbfa4784bu, 0xcbed606eu
};

constexpr bool matches_reference() {
   rt::Pcg32 rng(42, 54);
   for (const std::uint32_t expected : kReferenceOutputs) {
      if (rng.next_u32() != expected) {
         return false;
      }
   }
   return true;
}

constexpr int kSampleCount = 1'000'000;
} // namespace

TEST_CASE("conformité à l'implémentation de référence", "[rng]") {
   STATIC_REQUIRE(matches_reference());

   rt::Pcg32 rng(42, 54);
   for (const std::uint32_t expected : kReferenceOutputs) {
      CHECK(rng.next_u32() == expected);
   }
}

TEST_CASE("déterminisme : mêmes indices, même séquence", "[rng]") {
   rt::Pcg32 a = rt::Pcg32::make_pixel_rng(1234, 7);
   rt::Pcg32 b = rt::Pcg32::make_pixel_rng(1234, 7);

   for (int i = 0; i < 100; ++i) {
      REQUIRE(a.next_u32() == b.next_u32());
   }
}

TEST_CASE("indépendance entre pixels et entre échantillons", "[rng]") {
   const std::uint32_t reference = rt::Pcg32::make_pixel_rng(10, 0).next_u32();

   CHECK(rt::Pcg32::make_pixel_rng(11, 0).next_u32() != reference);
   CHECK(rt::Pcg32::make_pixel_rng(10, 1).next_u32() != reference);

   CHECK(rt::Pcg32::make_pixel_rng(0, 0).next_u32() != 0u);
}

TEST_CASE("next_float reste dans [0, 1[", "[rng]") {
   rt::Pcg32 rng = rt::Pcg32::make_pixel_rng(0, 0);

   for (int i = 0; i < kSampleCount; ++i) {
      const float u = rng.next_float();
      REQUIRE(u >= 0.0f);
      REQUIRE(u < 1.0f);
   }
}

TEST_CASE("next_float est uniforme : moyenne de 0,5", "[rng]") {
   rt::Pcg32 rng = rt::Pcg32::make_pixel_rng(42, 0);

   double sum = 0.0;
   for (int i = 0; i < kSampleCount; ++i) {
      sum += rng.next_float();
   }

   CHECK_THAT(sum / kSampleCount, Catch::Matchers::WithinAbs(0.5, 0.001));
}
