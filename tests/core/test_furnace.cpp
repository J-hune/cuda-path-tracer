#include "core/camera.h"
#include "core/environment.h"
#include "core/material.h"
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
#include <cstddef>
#include <cstdint>
#include <utility>

// Test du four blanc : un objet sans émission dans un environnement uniforme de radiance 1.
// Avec un albédo de 1, aucune énergie n'est absorbée : chaque pixel doit valoir exactement 1.
// L'objet est convexe : après un rebond, le rayon repart toujours vers l'environnement,
// donc chaque échantillon (et pas seulement la moyenne) vaut la valeur attendue.

namespace {

constexpr int kImageSize = 64;
constexpr int kSamplesPerPixel = 4;

// Cube [-1, 1]³ en 12 triangles, avec un matériau unique.
// Les triangles sont orientés vers l'extérieur, comme dans un glTF (nécessaire pour le verre).
rt::Scene make_cube(const rt::Material& material) {
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
   for (std::size_t i = 0; i < scene.indices.size(); i += 3) {
      const rt::Vec3 v0 = scene.positions[scene.indices[i]];
      const rt::Vec3 v1 = scene.positions[scene.indices[i + 1]];
      const rt::Vec3 v2 = scene.positions[scene.indices[i + 2]];
      // Le cube est centré sur l'origine : la normale extérieure va dans le sens du centre de la
      // face
      if (rt::dot(rt::cross(v1 - v0, v2 - v0), v0 + v1 + v2) < 0.0f) {
         std::swap(scene.indices[i + 1], scene.indices[i + 2]);
      }
   }
   scene.materials = {material};
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
   const rt::Scene scene = make_cube({.albedo = {albedo, albedo, albedo}, .emission = {0, 0, 0}});
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

            std::uint32_t ray_count = 0;
            const rt::Vec3 radiance = rt::trace_path(ray, view, rng, ray_count);
            // Cube convexe et aucune lampe : un rayon s'il rate le cube, deux s'il le touche
            REQUIRE(ray_count == (on_cube ? 2u : 1u));
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

TEST_CASE("four blanc : cube de verre invisible en moyenne", "[furnace]") {
   // Le verre ne fait que réfléchir ou réfracter : sans absorption, il conserve toute l'énergie.
   // Les rebonds internes et la roulette russe rendent chaque échantillon aléatoire :
   // seule la moyenne doit valoir 1.
   const rt::Scene scene = make_cube({
      .albedo = {1, 1, 1},
      .emission = {0, 0, 0},
      .transmission = 1.0f,
      .ior = 1.5f,
   });
   const rt::Bvh bvh = rt::build_bvh(scene);
   const rt::SceneView view = make_view(scene, bvh);
   const rt::Camera camera(
      {.position = {0, 0, 3}, .target = {0, 0, 0}, .up = {0, 1, 0}, .vertical_fov_degrees = 30},
      kImageSize,
      kImageSize
   );

   constexpr int kGlassSamples = 16;
   double sum = 0.0;
   int count = 0;
   for (int y = 0; y < kImageSize; ++y) {
      for (int x = 0; x < kImageSize; ++x) {
         const auto pixel = static_cast<std::uint32_t>((y * kImageSize) + x);
         for (int sample = 0; sample < kGlassSamples; ++sample) {
            rt::Pcg32 rng = rt::Pcg32::make_pixel_rng(pixel, static_cast<std::uint32_t>(sample));
            const rt::Ray ray = camera.generate_ray(
               static_cast<float>(x) + rng.next_float(),
               static_cast<float>(y) + rng.next_float()
            );
            std::uint32_t ray_count = 0;
            sum += rt::trace_path(ray, view, rng, ray_count).x;
            ++count;
         }
      }
   }

   CHECK_THAT(sum / count, Catch::Matchers::WithinAbs(1.0, 0.01));
}
