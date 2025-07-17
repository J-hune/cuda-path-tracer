#include "core/dielectric.h"
#include "core/rng.h"
#include "core/vec3.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

using Catch::Matchers::WithinAbs;

namespace {
constexpr float kGlassIor = 1.5f;
constexpr rt::Vec3 kUp{0, 1, 0}; // normale extérieure d'une surface horizontale
} // namespace

TEST_CASE("Fresnel : 4 % à incidence normale, 100 % en rasant", "[dielectric]") {
   CHECK_THAT(rt::fresnel_schlick(1.0f, kGlassIor), WithinAbs(0.04, 1e-6));
   CHECK_THAT(rt::fresnel_schlick(0.0f, kGlassIor), WithinAbs(1.0, 1e-6));
}

TEST_CASE("reflect : miroir autour de la normale", "[dielectric]") {
   STATIC_REQUIRE(rt::reflect({1, -1, 0}, kUp) == rt::Vec3{1, 1, 0});
}

TEST_CASE("Beer-Lambert : exp(-σ d)", "[dielectric]") {
   const rt::Vec3 t = rt::beer_lambert({1.0f, 0.0f, 2.0f}, 0.5f);
   CHECK_THAT(t.x, WithinAbs(std::exp(-0.5), 1e-6));
   CHECK_THAT(t.y, WithinAbs(1.0, 1e-6));
   CHECK_THAT(t.z, WithinAbs(std::exp(-1.0), 1e-6));
}

TEST_CASE("réfraction : loi de Snell en entrant dans le verre", "[dielectric]") {
   // Incidence à 45° depuis l'air : sin(θt) = sin(45°) / 1,5
   const rt::Vec3 d = rt::normalize({1, -1, 0});
   rt::Pcg32 rng(1, 0);
   int transmitted = 0;
   for (int i = 0; i < 1000; ++i) {
      const rt::DielectricSample s = rt::sample_dielectric(d, kUp, kGlassIor, rng);
      REQUIRE_THAT(rt::length(s.direction), WithinAbs(1.0, 1e-5));
      if (s.transmitted) {
         ++transmitted;
         REQUIRE(s.direction.y < 0.0f); // passe sous la surface
         REQUIRE_THAT(s.direction.x, WithinAbs(std::sin(std::acos(-1.0) / 4) / kGlassIor, 1e-5));
      } else {
         REQUIRE(s.direction == rt::reflect(d, kUp));
      }
   }
   // À 45°, Fresnel réfléchit environ 5 % : la grande majorité est transmise
   CHECK(transmitted > 900);
}

TEST_CASE("réflexion totale interne en sortant du verre", "[dielectric]") {
   // Depuis l'intérieur, au-delà de l'angle critique (41,8° pour 1,5) : toujours réfléchi
   const rt::Vec3 d = rt::normalize({1, 0.5f, 0}); // 63° de la normale, vers l'extérieur
   rt::Pcg32 rng(2, 0);
   for (int i = 0; i < 100; ++i) {
      const rt::DielectricSample s = rt::sample_dielectric(d, kUp, kGlassIor, rng);
      REQUIRE_FALSE(s.transmitted);
      REQUIRE(s.direction.y < 0.0f); // reste dans le verre
   }
}
