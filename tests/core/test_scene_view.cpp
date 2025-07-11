#include "core/ray.h"
#include "core/rng.h"
#include "core/scene_view.h"
#include "core/triangle.h"
#include "core/vec3.h"
#include "scene/bvh_builder.h"
#include "scene/gltf_loader.h"

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>

namespace {

constexpr std::uint32_t kTriangleCount = 200;
constexpr int kRayCount = 1000;

rt::Vec3 random_point(rt::Pcg32& rng, float scale) {
   return {
      (rng.next_float() - 0.5f) * scale,
      (rng.next_float() - 0.5f) * scale,
      (rng.next_float() - 0.5f) * scale
   };
}

// Triangles de taille aléatoire, dispersés dans un cube de côté 10
rt::Scene make_random_scene(rt::Pcg32& rng) {
   rt::Scene scene;
   for (std::uint32_t i = 0; i < kTriangleCount; ++i) {
      const rt::Vec3 center = random_point(rng, 10.f);
      for (std::uint32_t corner = 0; corner < 3; ++corner) {
         scene.indices.push_back(static_cast<std::uint32_t>(scene.positions.size()));
         scene.positions.push_back(center + random_point(rng, 2.f));
      }
   }
   return scene;
}

// Référence : teste tous les triangles un par un
bool intersect_brute_force(rt::Ray ray, const rt::Scene& scene, rt::Hit& hit) {
   hit.t = rt::kInfinity;
   for (std::uint32_t triangle = 0; triangle < kTriangleCount; ++triangle) {
      const std::size_t base = 3 * static_cast<std::size_t>(triangle);
      rt::TriangleHit triangle_hit{};
      if (rt::intersect_triangle(
             ray,
             scene.positions[scene.indices[base]],
             scene.positions[scene.indices[base + 1]],
             scene.positions[scene.indices[base + 2]],
             triangle_hit
          ) &&
          triangle_hit.t < hit.t) {
         hit = {triangle_hit.t, triangle, triangle_hit.u, triangle_hit.v};
      }
   }
   return hit.t < rt::kInfinity;
}

} // namespace

TEST_CASE("intersect_scene trouve la même intersection que la force brute", "[scene_view]") {
   rt::Pcg32 rng(2024, 1);
   const rt::Scene scene = make_random_scene(rng);
   const rt::Bvh bvh = rt::build_bvh(scene);
   const rt::SceneView view{
      .nodes = bvh.nodes.data(),
      .triangle_indices = bvh.triangle_indices.data(),
      .positions = scene.positions.data(),
      .indices = scene.indices.data()
   };

   int hit_count = 0;
   for (int i = 0; i < kRayCount; ++i) {
      // Rayons visant le cœur de la scène, pour qu'une bonne partie touche un triangle
      const rt::Vec3 origin = random_point(rng, 20.f);
      const rt::Ray ray{origin, rt::normalize(random_point(rng, 6.f) - origin)};

      rt::Hit expected{};
      rt::Hit actual{};
      const bool expected_hit = intersect_brute_force(ray, scene, expected);
      const bool actual_hit = rt::intersect_scene(ray, view, actual);

      REQUIRE(actual_hit == expected_hit);
      if (expected_hit) {
         ++hit_count;
         REQUIRE(actual.triangle == expected.triangle);
         REQUIRE(actual.t == expected.t);
      }
   }

   // Le test n'a de sens que si une bonne partie des rayons touche quelque chose
   CHECK(hit_count > kRayCount / 10);
}

TEST_CASE("intersect_scene sur une scène vide", "[scene_view]") {
   const rt::SceneView view{
      .nodes = nullptr, .triangle_indices = nullptr, .positions = nullptr, .indices = nullptr
   };
   rt::Hit hit{};
   CHECK_FALSE(rt::intersect_scene({{0, 0, 0}, {0, 0, -1}}, view, hit));
}
