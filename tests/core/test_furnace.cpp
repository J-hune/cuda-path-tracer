#include "core/camera.h"
#include "core/environment.h"
#include "core/path_tracer.h"
#include "core/ray.h"
#include "core/rng.h"
#include "core/scene_view.h"
#include "core/vec3.h"
#include "scene/bvh_builder.h"
#include "scene/gltf_loader.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdint>

// Test du four blanc : un objet sans émission dans un environnement uniforme de radiance 1.
// Avec un albédo de 1, aucune énergie n'est absorbée : chaque pixel doit valoir exactement 1.
// L'objet est convexe : après un rebond, le rayon repart toujours vers l'environnement,
// donc chaque échantillon (et pas seulement la moyenne) vaut la valeur attendue.

namespace {

constexpr int kImageSize = 64;
constexpr int kSamplesPerPixel = 4;

// Cube [-1, 1]³ en 12 triangles, avec un matériau unique
rt::Scene make_cube(rt::Vec3 albedo) {
   rt::Scene scene;
   scene.positions = {
      {-1, -1, -1},
      {1, -1, -1},
      {1, 1, -1},
      {-1, 1, -1},
      {-1, -1, 1},
      {1, -1, 1},
      {1, 1, 1},
      {-1, 1, 1},
   };
   scene.indices = {0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1,
                    3, 2, 6, 3, 6, 7, 0, 3, 7, 0, 7, 4, 1, 5, 6, 1, 6, 2};
   scene.materials = {{.albedo = albedo, .emission = {0, 0, 0}}};
   scene.material_indices.assign(12, 0);
   scene.environment = {.bottom = {1, 1, 1}, .top = {1, 1, 1}};
   return scene;
}

rt::SceneView make_view(const rt::Scene& scene, const rt::Bvh& bvh) {
   return {
      .nodes = bvh.nodes.data(),
      .triangle_indices = bvh.triangle_indices.data(),
      .positions = scene.positions.data(),
      .indices = scene.indices.data(),
      .materials = scene.materials.data(),
      .material_indices = scene.material_indices.data(),
      .emissive_triangles = nullptr,
      .emissive_triangle_count = 0,
      .environment = scene.environment,
   };
}

// Rend l'image et vérifie chaque échantillon : expected sur le cube, 1 à côté
void check_furnace(float albedo, float expected_on_cube) {
   const rt::Scene scene = make_cube({albedo, albedo, albedo});
   const rt::Bvh bvh = rt::build_bvh(scene);
   const rt::SceneView view = make_view(scene, bvh);
   const rt::Camera camera(
      {.position = {2, 1.5f, 4}, .target = {0, 0, 0}, .up = {0, 1, 0}, .vertical_fov_degrees = 50},
      kImageSize,
      kImageSize
   );

   int hits = 0;
   for (int y = 0; y < kImageSize; ++y) {
      for (int x = 0; x < kImageSize; ++x) {
         const auto pixel = static_cast<std::uint32_t>((y * kImageSize) + x);
         for (int sample = 0; sample < kSamplesPerPixel; ++sample) {
            rt::Pcg32 rng = rt::Pcg32::make_pixel_rng(pixel, static_cast<std::uint32_t>(sample));
            const rt::Ray ray = camera.generate_ray(
               static_cast<float>(x) + rng.next_float(),
               static_cast<float>(y) + rng.next_float()
            );

            rt::Hit hit{};
            const bool on_cube = rt::intersect_scene(ray, view, hit);
            hits += on_cube ? 1 : 0;
            const float expected = on_cube ? expected_on_cube : 1.0f;

            const rt::Vec3 radiance = rt::trace_path(ray, view, rng);
            REQUIRE_THAT(radiance.x, Catch::Matchers::WithinAbs(expected, 1e-5));
            REQUIRE_THAT(radiance.y, Catch::Matchers::WithinAbs(expected, 1e-5));
            REQUIRE_THAT(radiance.z, Catch::Matchers::WithinAbs(expected, 1e-5));
         }
      }
   }

   // Le cube doit occuper une bonne partie de l'image pour que le test ait du sens
   CHECK(hits > kImageSize * kImageSize * kSamplesPerPixel / 10);
}

} // namespace

TEST_CASE("four blanc : objet d'albédo 1 invisible", "[furnace]") {
   check_furnace(1.0f, 1.0f);
}

TEST_CASE("four blanc : objet d'albédo 0,5 renvoie la moitié de la lumière", "[furnace]") {
   check_furnace(0.5f, 0.5f);
}
