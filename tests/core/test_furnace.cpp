#include "core/bsdf.h"
#include "core/camera.h"
#include "core/environment.h"
#include "core/material.h"
#include "core/path_tracer.h"
#include "core/ray.h"
#include "core/rng.h"
#include "core/scene_view.h"
#include "core/vec3.h"
#include "scene/bvh_builder.h"
#include "scene/environment_map.h"
#include "scene/gltf_loader.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

// Test du four blanc : un objet sans émission dans un environnement uniforme de radiance 1.
// L'objet est convexe : après un rebond, le rayon repart toujours vers l'environnement. La lumière
// rapportée vaut donc, en moyenne, l'albédo directionnel de la BSDF : la part de lumière qu'elle
// renvoie pour cet angle de vue. Le test compare le rendu à cet albédo, calculé indépendamment en
// intégrant eval sur l'hémisphère.

namespace {

constexpr int kImageSize = 64;

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
      .bvh_triangles = bvh.triangles.data(),
      .positions = scene.positions.data(),
      .indices = scene.indices.data(),
      .materials = scene.materials.data(),
      .material_indices = scene.material_indices.data(),
      .emissive_triangles = nullptr,
      .emissive_triangle_count = 0,
      .environment = scene.environment,
      .environment_map = {},
      .caustics = {},
   };
}

// Albédo directionnel ∫ f(wo, wi) cos θi dωi, par la méthode du point milieu en (θi, φi)
double directional_albedo(const rt::SurfaceParams& params, rt::Vec3 wo) {
   constexpr int kThetaSteps = 600;
   constexpr int kPhiSteps = 300;
   double sum = 0.0;
   for (int i = 0; i < kThetaSteps; ++i) {
      const double theta = (i + 0.5) * (rt::kPi / 2.0) / kThetaSteps;
      for (int j = 0; j < kPhiSteps; ++j) {
         const double phi = (j + 0.5) * (2.0 * rt::kPi) / kPhiSteps;
         const rt::Vec3 wi{
            static_cast<float>(std::sin(theta) * std::cos(phi)),
            static_cast<float>(std::sin(theta) * std::sin(phi)),
            static_cast<float>(std::cos(theta))
         };
         sum += rt::principled::eval(params, wo, wi).x * wi.z * std::sin(theta) *
                (rt::kPi / 2.0 / kThetaSteps) * (2.0 * rt::kPi / kPhiSteps);
      }
   }
   return sum;
}

// Lance des rayons fixes vers le cube sous plusieurs angles, et compare la moyenne de la lumière
// rapportée à l'albédo directionnel attendu. Vérifie aussi les rayons qui ratent le cube.
void check_furnace(const rt::SurfaceParams& params) {
   const rt::Scene scene = make_cube({
      .albedo = params.base_color,
      .emission = {0, 0, 0},
      .metallic = params.metallic,
      .roughness = params.roughness,
   });
   const rt::Bvh bvh = rt::build_bvh(scene);
   const rt::SceneView view = make_view(scene, bvh);

   // Rayons visant la face z = 1 (normale +z) sous des angles de plus en plus rasants
   constexpr int kSamples = 40'000;
   for (const float angle_degrees : {0.0f, 30.0f, 60.0f, 80.0f}) {
      const float angle = angle_degrees * rt::kPi / 180.0f;
      const rt::Vec3 direction{std::sin(angle), 0.0f, -std::cos(angle)};
      const rt::Ray ray{rt::Vec3{0.0f, 0.2f, 1.0f} - (direction * 3.0f), direction};
      const rt::Vec3 wo = -direction; // dans le repère de la face, la normale est +z

      double sum = 0.0;
      for (int i = 0; i < kSamples; ++i) {
         rt::Pcg32 rng = rt::Pcg32::make_pixel_rng(0, static_cast<std::uint32_t>(i));
         std::uint32_t ray_count = 0;
         sum += rt::trace_path(ray, view, rng, ray_count).x;
         // Cube convexe et aucune lampe : le rayon caméra, puis au plus un rebond qui s'échappe
         REQUIRE(ray_count <= 2u);
      }
      const double expected = directional_albedo(params, wo);
      INFO("angle = " << angle_degrees << "°, albédo attendu = " << expected);
      CHECK_THAT(sum / kSamples, Catch::Matchers::WithinRel(expected, 0.01));
   }

   // Un rayon qui rate le cube voit l'environnement : exactement 1, avec un seul rayon
   const rt::Ray miss{{5.0f, 5.0f, 5.0f}, {0.0f, 1.0f, 0.0f}};
   rt::Pcg32 rng(1, 0);
   std::uint32_t ray_count = 0;
   const rt::Vec3 radiance = rt::trace_path(miss, view, rng, ray_count);
   CHECK(radiance == rt::Vec3{1.0f, 1.0f, 1.0f});
   CHECK(ray_count == 1u);
}

} // namespace

TEST_CASE("four blanc : diélectrique blanc rugueux", "[furnace]") {
   check_furnace({.base_color = {1, 1, 1}, .metallic = 0.0f, .roughness = 1.0f});
}

TEST_CASE("four blanc : diélectrique gris semi-brillant", "[furnace]") {
   check_furnace({.base_color = {0.5f, 0.5f, 0.5f}, .metallic = 0.0f, .roughness = 0.4f});
}

TEST_CASE("four blanc : métal blanc rugueux", "[furnace]") {
   check_furnace({.base_color = {1, 1, 1}, .metallic = 1.0f, .roughness = 0.6f});
}

namespace {

// Moyenne de la lumière vue à travers un cube de verre, dans l'environnement uniforme de radiance 1
double glass_cube_average(float roughness) {
   const rt::Scene scene = make_cube({
      .albedo = {1, 1, 1},
      .emission = {0, 0, 0},
      .roughness = roughness,
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
   return sum / count;
}

} // namespace

TEST_CASE("four blanc : cube de verre invisible en moyenne", "[furnace]") {
   // Le verre ne fait que réfléchir ou réfracter : sans absorption, il conserve toute l'énergie.
   // Les rebonds internes et la roulette russe rendent chaque échantillon aléatoire :
   // seule la moyenne doit valoir 1.
   CHECK_THAT(glass_cube_average(0.0f), Catch::Matchers::WithinAbs(1.0, 0.01));
}

TEST_CASE("four blanc : un cube de verre peu dépoli ne crée pas de lumière", "[furnace]") {
   // Une seule diffusion par microfacette : un peu d'énergie se perd, aucune n'est créée
   const double average = glass_cube_average(0.15f);
   CHECK(average <= 1.01);
   CHECK(average > 0.95);
}

TEST_CASE("four blanc : une HDRI uniforme éclaire comme l'environnement uniforme", "[furnace]") {
   // HDRI blanche de luminance 1 : tirée par l'éclairage direct et pondérée par MIS
   rt::EnvironmentImage image;
   image.width = 16;
   image.height = 8;
   image.texels.assign(std::size_t{16} * 8, rt::Vec3{1.0f, 1.0f, 1.0f});
   rt::build_environment_distribution(image);

   const rt::SurfaceParams params{
      .base_color = {0.8f, 0.8f, 0.8f},
      .metallic = 0.0f,
      .roughness = 0.5f
   };
   rt::Scene scene = make_cube({
      .albedo = params.base_color,
      .emission = {0, 0, 0},
      .metallic = params.metallic,
      .roughness = params.roughness,
   });
   scene.environment = {};
   const rt::Bvh bvh = rt::build_bvh(scene);
   rt::SceneView view = make_view(scene, bvh);
   view.environment_map = image.view();

   constexpr int kSamples = 40'000;
   for (const float angle_degrees : {0.0f, 60.0f}) {
      const float angle = angle_degrees * rt::kPi / 180.0f;
      const rt::Vec3 direction{std::sin(angle), 0.0f, -std::cos(angle)};
      const rt::Ray ray{rt::Vec3{0.0f, 0.2f, 1.0f} - (direction * 3.0f), direction};
      double sum = 0.0;
      for (int i = 0; i < kSamples; ++i) {
         rt::Pcg32 rng = rt::Pcg32::make_pixel_rng(0, static_cast<std::uint32_t>(i));
         std::uint32_t ray_count = 0;
         sum += rt::trace_path(ray, view, rng, ray_count).x;
      }
      INFO("angle = " << angle_degrees << "°");
      CHECK_THAT(
         sum / kSamples,
         Catch::Matchers::WithinRel(directional_albedo(params, -direction), 0.01)
      );
   }
}
