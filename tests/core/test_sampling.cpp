#include "core/rng.h"
#include "core/sampling.h"
#include "core/vec3.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace {
constexpr int kSampleCount = 100'000;
}

TEST_CASE("random_unit_vector : longueur 1 et moyenne nulle", "[sampling]") {
   rt::Pcg32 rng(7, 0);
   double sum_x = 0.0;
   double sum_y = 0.0;
   double sum_z = 0.0;

   for (int i = 0; i < kSampleCount; ++i) {
      const rt::Vec3 v = rt::random_unit_vector(rng);
      REQUIRE_THAT(rt::length(v), Catch::Matchers::WithinAbs(1.0, 1e-5));
      sum_x += v.x;
      sum_y += v.y;
      sum_z += v.z;
   }

   CHECK_THAT(sum_x / kSampleCount, Catch::Matchers::WithinAbs(0.0, 0.01));
   CHECK_THAT(sum_y / kSampleCount, Catch::Matchers::WithinAbs(0.0, 0.01));
   CHECK_THAT(sum_z / kSampleCount, Catch::Matchers::WithinAbs(0.0, 0.01));
}

TEST_CASE("sample_cosine_hemisphere : directions dans l'hémisphère de la normale", "[sampling]") {
   rt::Pcg32 rng(7, 1);
   const rt::Vec3 normal = rt::normalize({1, 2, -3});

   for (int i = 0; i < kSampleCount; ++i) {
      const rt::Vec3 direction = rt::sample_cosine_hemisphere(normal, rng);
      REQUIRE_THAT(rt::length(direction), Catch::Matchers::WithinAbs(1.0, 1e-5));
      REQUIRE(rt::dot(direction, normal) >= 0.f);
   }
}

TEST_CASE("sample_cosine_hemisphere : répartition en cosinus", "[sampling]") {
   // Pour une densité en cos(θ)/π, l'espérance de cos(θ) vaut 2/3
   rt::Pcg32 rng(7, 2);
   const rt::Vec3 normal{0, 0, 1};

   double sum_cos = 0.0;
   for (int i = 0; i < kSampleCount; ++i) {
      sum_cos += rt::dot(rt::sample_cosine_hemisphere(normal, rng), normal);
   }

   CHECK_THAT(sum_cos / kSampleCount, Catch::Matchers::WithinAbs(2.0 / 3.0, 0.005));
}
