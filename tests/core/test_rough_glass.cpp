#include "core/dielectric.h"
#include "core/light.h"
#include "core/path_tracer.h"
#include "core/ray.h"
#include "core/rng.h"
#include "core/scene_view.h"
#include "core/vec2.h"
#include "core/vec3.h"
#include "scene/bvh_builder.h"
#include "scene/gltf_loader.h"
#include "scene/host_view.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>

// Une plaque de verre dépoli horizontale z = 0 (une seule face, normale +z), une lampe en dessous,
// et un rayon caméra vertical qui touche la plaque par-dessus : la lumière de la lampe n'arrive
// qu'en traversant la plaque.

namespace {

constexpr float kRoughness = 0.4f;
constexpr float kIor = 1.5f;
constexpr rt::Vec3 kTint{0.9f, 0.8f, 0.7f};

void add_quad(rt::Scene& s, float half, float z, std::uint32_t material) {
   const auto base = static_cast<std::uint32_t>(s.positions.size());
   s.positions.insert(
      s.positions.end(),
      {{-half, -half, z}, {half, -half, z}, {half, half, z}, {-half, half, z}}
   );
   s.uvs.insert(s.uvs.end(), 4, rt::Vec2{0, 0});
   s.indices.insert(s.indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
   s.material_indices.insert(s.material_indices.end(), 2, material);
}

rt::Scene make_plate() {
   rt::Scene s;
   s.materials = {
      {.albedo = kTint,
       .emission = {0, 0, 0},
       .roughness = kRoughness,
       .transmission = 1.0f,
       .ior = kIor},
      {.albedo = {0, 0, 0}, .emission = {4, 4, 4}},
   };
   add_quad(s, 10.0f, 0.0f, 0);
   return s;
}

constexpr rt::Ray kCameraRay{{0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f}};

} // namespace

TEST_CASE("verre dépoli : lampe ponctuelle vue à travers la plaque", "[rough_glass]") {
   rt::Scene s = make_plate();
   const rt::PunctualLight light{
      .type = rt::PunctualLight::Type::Point,
      .position = {0.7f, -0.3f, -1.5f},
      .intensity = {10.0f, 10.0f, 10.0f},
   };
   s.punctual_lights = {light};
   const rt::Bvh bvh = rt::build_bvh(s);
   const rt::SceneView view = rt::host_view(s, bvh);

   // La lampe est la seule lumière et le ciel est noir : chaque échantillon vaut f · E · |cos θ|
   const rt::Vec3 to_light = light.position;
   const rt::Vec3 wi = rt::normalize(to_light);
   const float f = rt::rough_dielectric::eval({0, 0, 1}, wi, kRoughness * kRoughness, kIor);
   const rt::Vec3 expected = kTint * light.intensity *
                             (f * std::fabs(wi.z) / rt::length_squared(to_light));
   REQUIRE(rt::max_component(expected) > 1e-3f);
   for (int i = 0; i < 16; ++i) {
      rt::Pcg32 rng = rt::Pcg32::make_pixel_rng(0, static_cast<std::uint32_t>(i));
      std::uint32_t ray_count = 0;
      const rt::Vec3 radiance = rt::trace_path(kCameraRay, view, rng, ray_count);
      INFO("obtenu " << radiance.x << ", attendu " << expected.x);
      CHECK_THAT(radiance.x, Catch::Matchers::WithinRel(expected.x, 1e-4f));
      CHECK_THAT(radiance.z, Catch::Matchers::WithinRel(expected.z, 1e-4f));
   }
}

TEST_CASE(
   "verre dépoli : même moyenne avec ou sans MIS sous une lampe surfacique",
   "[rough_glass]"
) {
   rt::Scene s = make_plate();
   add_quad(s, 0.5f, -1.0f, 1);
   s.emissive_triangles = {2, 3};
   const rt::Bvh bvh = rt::build_bvh(s);
   rt::SceneView view = rt::host_view(s, bvh);

   constexpr int kSamples = 200'000;
   const auto average = [&](bool mis) {
      view.use_mis = mis;
      rt::Vec3 sum{0.0f, 0.0f, 0.0f};
      for (int i = 0; i < kSamples; ++i) {
         rt::Pcg32 rng = rt::Pcg32::make_pixel_rng(1, static_cast<std::uint32_t>(i));
         std::uint32_t ray_count = 0;
         sum += rt::trace_path(kCameraRay, view, rng, ray_count);
      }
      return sum / static_cast<float>(kSamples);
   };
   // Sans MIS, seul l'éclairage direct compte la lampe ; avec MIS, les deux stratégies
   const rt::Vec3 direct_only = average(false);
   const rt::Vec3 combined = average(true);
   INFO("sans MIS " << direct_only.x << ", avec MIS " << combined.x);
   REQUIRE(direct_only.x > 0.01f);
   CHECK_THAT(combined.x, Catch::Matchers::WithinRel(direct_only.x, 0.01f));
}
