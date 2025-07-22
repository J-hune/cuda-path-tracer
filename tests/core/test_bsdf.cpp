#include "core/bsdf.h"
#include "core/rng.h"
#include "core/sampling.h"
#include "core/vec3.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

using Catch::Matchers::WithinAbs;

namespace {
constexpr int kSampleCount = 200'000;
constexpr rt::Vec3 kAlbedo{0.8f, 0.5f, 0.2f};
// Direction d'observation quelconque, au-dessus de la surface
rt::Vec3 outgoing() {
   return rt::normalize({0.3f, -0.2f, 0.9f});
}
} // namespace

TEST_CASE("Lambert : sample, eval et pdf sont cohérents", "[bsdf]") {
   rt::Pcg32 rng(5, 0);
   for (int i = 0; i < 10'000; ++i) {
      const rt::BsdfSample s = rt::lambert::sample(
         kAlbedo,
         outgoing(),
         {rng.next_float(), rng.next_float()}
      );
      if (s.pdf <= 0.0f) {
         continue;
      }
      // La densité renvoyée par sample est celle de pdf
      REQUIRE_THAT(s.pdf, WithinAbs(rt::lambert::pdf(outgoing(), s.wi), 1e-5));
      // Le poids vaut eval · cos θi / pdf
      const rt::Vec3 expected = rt::lambert::eval(kAlbedo, outgoing(), s.wi) * (s.wi.z / s.pdf);
      REQUIRE_THAT(s.weight.x, WithinAbs(expected.x, 1e-4));
      REQUIRE_THAT(s.weight.y, WithinAbs(expected.y, 1e-4));
      REQUIRE_THAT(s.weight.z, WithinAbs(expected.z, 1e-4));
   }
}

TEST_CASE("Lambert : la densité s'intègre à 1 sur la sphère", "[bsdf]") {
   // ∫ pdf dω estimé avec des directions uniformes sur la sphère (densité 1 / 4π)
   rt::Pcg32 rng(5, 1);
   double sum = 0.0;
   for (int i = 0; i < kSampleCount; ++i) {
      sum += rt::lambert::pdf(outgoing(), rt::random_unit_vector(rng)) * 4.0 * rt::kPi;
   }
   CHECK_THAT(sum / kSampleCount, WithinAbs(1.0, 0.01));
}

TEST_CASE("Lambert : conservation de l'énergie et réciprocité", "[bsdf]") {
   // Albédo intégré : ∫ f cos θi dω = albedo, jamais plus que 1
   rt::Pcg32 rng(5, 2);
   double sum = 0.0;
   for (int i = 0; i < kSampleCount; ++i) {
      const rt::Vec3 wi = rt::random_unit_vector(rng);
      sum += rt::lambert::eval({1, 1, 1}, outgoing(), wi).x * std::fmax(wi.z, 0.0f) * 4.0 * rt::kPi;
   }
   CHECK_THAT(sum / kSampleCount, WithinAbs(1.0, 0.01));

   // Réciprocité de Helmholtz : f(wo, wi) = f(wi, wo)
   const rt::Vec3 wi = rt::normalize({-0.5f, 0.1f, 0.4f});
   CHECK(rt::lambert::eval(kAlbedo, outgoing(), wi) == rt::lambert::eval(kAlbedo, wi, outgoing()));
}
