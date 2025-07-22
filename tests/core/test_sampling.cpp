#include "core/rng.h"
#include "core/sampling.h"
#include "core/vec2.h"
#include "core/vec3.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

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

TEST_CASE("mapping concentrique : points remarquables", "[sampling]") {
   const auto check = [](rt::Vec2 u, rt::Vec2 expected) {
      const rt::Vec2 d = rt::square_to_disk_concentric(u);
      CHECK_THAT(d.x, Catch::Matchers::WithinAbs(expected.x, 1e-6));
      CHECK_THAT(d.y, Catch::Matchers::WithinAbs(expected.y, 1e-6));
   };
   const float diagonal = std::sqrt(0.5f);
   check({0.5f, 0.5f}, {0.0f, 0.0f}); // centre du carré -> centre du disque
   check({1.0f, 0.5f}, {1.0f, 0.0f}); // milieu d'un bord -> bord du disque
   check({0.5f, 0.0f}, {0.0f, -1.0f});
   check({1.0f, 1.0f}, {diagonal, diagonal}); // coin -> bord du disque, sur la diagonale
   check({0.0f, 0.0f}, {-diagonal, -diagonal});
}

TEST_CASE("mapping concentrique : uniforme dans le disque", "[sampling]") {
   // Préserve les aires : un quart des points dans le disque de rayon 1/2, aucun hors du disque
   rt::Pcg32 rng(7, 3);
   int inside_half = 0;
   for (int i = 0; i < kSampleCount; ++i) {
      const rt::Vec2 d = rt::square_to_disk_concentric({rng.next_float(), rng.next_float()});
      const float r2 = (d.x * d.x) + (d.y * d.y);
      REQUIRE(r2 <= 1.0f + 1e-6f);
      inside_half += r2 < 0.25f ? 1 : 0;
   }
   CHECK_THAT(
      static_cast<double>(inside_half) / kSampleCount,
      Catch::Matchers::WithinAbs(0.25, 0.005)
   );
}

TEST_CASE(
   "hémisphère en cosinus : directions unitaires vers +z, répartition en cosinus",
   "[sampling]"
) {
   // Pour une densité cos θ / π, l'espérance de cos θ vaut 2/3 (elle vaudrait 1/2 en uniforme)
   rt::Pcg32 rng(7, 2);
   double sum_cos = 0.0;
   for (int i = 0; i < kSampleCount; ++i) {
      const rt::Vec3 d = rt::square_to_cosine_hemisphere({rng.next_float(), rng.next_float()});
      REQUIRE_THAT(rt::length(d), Catch::Matchers::WithinAbs(1.0, 1e-5));
      REQUIRE(d.z >= 0.0f);
      sum_cos += d.z;
   }
   CHECK_THAT(sum_cos / kSampleCount, Catch::Matchers::WithinAbs(2.0 / 3.0, 0.005));
}
