#include "core/path_tracer.h"
#include "core/rng.h"
#include "core/scene_view.h"
#include "scene/gltf_loader.h"
#include "scene/light_distribution.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

// Carré 1 × 1 dans le plan z = height, uv de (0, 0) à (1, 1)
void add_quad(rt::Scene& s, float height, std::uint32_t material) {
   const auto base = static_cast<std::uint32_t>(s.positions.size());
   s.positions.insert(
      s.positions.end(),
      {{0, 0, height}, {1, 0, height}, {1, 1, height}, {0, 1, height}}
   );
   s.uvs.insert(s.uvs.end(), {{0, 0}, {1, 0}, {1, 1}, {0, 1}});
   s.indices.insert(s.indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
   s.material_indices.insert(s.material_indices.end(), 2, material);
}

} // namespace

TEST_CASE(
   "répartition des lumières : proportionnelle à la puissance, texture d'émission comprise",
   "[lights]"
) {
   rt::Scene s;
   // Texture 2 × 2 : moitié gauche (u < 0,5) blanche, moitié droite noire
   s.textures = {{.offset = 0, .width = 2, .height = 2}};
   s.texels = {{1, 1, 1}, {0, 0, 0}, {1, 1, 1}, {0, 0, 0}};
   s.materials = {
      {.albedo = {0, 0, 0}, .emission = {1, 1, 1}},
      {.albedo = {0, 0, 0}, .emission = {3, 3, 3}},
      {.albedo = {0, 0, 0}, .emission = {2, 2, 2}, .emissive_texture = 0},
      {.albedo = {0.5f, 0.5f, 0.5f}, .emission = {0, 0, 0}},
   };
   add_quad(s, 0.0f, 0);
   add_quad(s, 1.0f, 1);
   add_quad(s, 2.0f, 2);
   add_quad(s, 3.0f, 3);
   s.emissive_triangles = {0, 1, 2, 3, 4, 5};
   const rt::LightDistribution lights = rt::light_distribution(s);

   REQUIRE(lights.cdf.size() == 7);
   const auto probability = [&](std::size_t k) { return lights.cdf[k + 1] - lights.cdf[k]; };
   // Puissances par triangle : 1, 1, 3, 3, puis le carré texturé. Son premier triangle (sommets
   // 0, 1, 2) est surtout à droite, sur le noir ; le second (0, 2, 3) surtout à gauche, sur le
   // blanc.
   CHECK_THAT(probability(0), WithinRel(probability(1), 1e-5f));
   CHECK_THAT(probability(2), WithinRel(3.0f * probability(0), 1e-5f));
   CHECK(probability(4) < probability(5));
   CHECK_THAT(lights.cdf.back(), WithinAbs(1.0, 1e-7));
   // Index de lumière de chaque triangle, -1 pour le carré non émissif
   CHECK(lights.triangle_light[2] == 2);
   CHECK(lights.triangle_light[6] == -1);
}

TEST_CASE(
   "choix des lumières : fréquences de la répartition, jamais une lumière de puissance nulle",
   "[lights]"
) {
   const std::array<float, 5> cdf{0.0f, 0.1f, 0.1f, 0.7f, 1.0f};
   rt::SceneView view;
   view.punctual_light_count = 4;
   view.light_cdf = cdf.data();
   std::array<int, 4> counts{};
   rt::Pcg32 rng(3, 0);
   constexpr int kDraws = 100'000;
   for (int i = 0; i < kDraws; ++i) {
      const rt::LightChoice choice = rt::pick_light(view, rng.next_float());
      ++counts[choice.index];
      CHECK_THAT(choice.probability, WithinAbs(cdf[choice.index + 1] - cdf[choice.index], 1e-7));
   }
   CHECK(counts[1] == 0);
   CHECK_THAT(counts[0] / static_cast<double>(kDraws), WithinAbs(0.1, 0.005));
   CHECK_THAT(counts[2] / static_cast<double>(kDraws), WithinAbs(0.6, 0.005));
   CHECK_THAT(counts[3] / static_cast<double>(kDraws), WithinAbs(0.3, 0.005));
}
