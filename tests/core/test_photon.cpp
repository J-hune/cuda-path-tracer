#include "core/light.h"
#include "core/photon_tracer.h"
#include "core/rng.h"
#include "core/scene_view.h"
#include "core/vec2.h"
#include "core/vec3.h"
#include "scene/bvh_builder.h"
#include "scene/gltf_loader.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>
#include <vector>

using Catch::Matchers::WithinRel;

namespace {

constexpr std::uint32_t kFloorMaterial = 0;
constexpr std::uint32_t kGlassMaterial = 1;
constexpr std::uint32_t kMirrorMaterial = 2;

struct TestScene {
   rt::Scene scene;
   rt::Bvh bvh;

   [[nodiscard]] rt::SceneView view() const {
      rt::SceneView view;
      view.nodes = bvh.nodes.data();
      view.triangle_indices = bvh.triangle_indices.data();
      view.positions = scene.positions.data();
      view.indices = scene.indices.data();
      view.materials = scene.materials.data();
      view.material_indices = scene.material_indices.data();
      view.punctual_lights = scene.punctual_lights.data();
      view.punctual_light_count = static_cast<std::uint32_t>(scene.punctual_lights.size());
      view.uvs = scene.uvs.data();
      return view;
   }
};

// Carré horizontal z = height, normale vers le haut (up) ou vers le bas
void add_quad(rt::Scene& s, float half, float height, bool up, std::uint32_t material) {
   const auto base = static_cast<std::uint32_t>(s.positions.size());
   s.positions.insert(
      s.positions.end(),
      {{-half, -half, height}, {half, -half, height}, {half, half, height}, {-half, half, height}}
   );
   s.uvs.insert(s.uvs.end(), 4, rt::Vec2{0, 0});
   if (up) {
      s.indices.insert(s.indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
   } else {
      s.indices.insert(s.indices.end(), {base, base + 2, base + 1, base, base + 3, base + 2});
   }
   s.material_indices.insert(s.material_indices.end(), 2, material);
}

// Sol mat z = 0, plus une lame de verre d'indice 1 (aucune réflexion ni déviation) ou un miroir
TestScene make_scene(const rt::PunctualLight& light, bool glass, bool mirror) {
   TestScene t;
   rt::Scene& s = t.scene;
   s.materials = {
      {.albedo = {0.8f, 0.8f, 0.8f}, .emission = {0, 0, 0}},
      {.albedo = {1, 1, 1},
       .emission = {0, 0, 0},
       .roughness = 0.0f,
       .transmission = 1.0f,
       .ior = 1.0f},
      {.albedo = {1, 1, 1}, .emission = {0, 0, 0}, .metallic = 1.0f, .roughness = 0.0f},
   };
   add_quad(s, 10.0f, 0.0f, true, kFloorMaterial);
   if (glass) {
      add_quad(s, 2.0f, 1.1f, true, kGlassMaterial);
      add_quad(s, 2.0f, 1.0f, false, kGlassMaterial);
   }
   if (mirror) {
      add_quad(s, 5.0f, 2.0f, false, kMirrorMaterial);
   }
   s.punctual_lights = {light};
   t.bvh = rt::build_bvh(s);
   return t;
}

// Spot étroit dirigé selon direction, pleine intensité au centre du cône
rt::PunctualLight spot(rt::Vec3 position, rt::Vec3 direction) {
   return {
      .type = rt::PunctualLight::Type::Spot,
      .position = position,
      .direction = direction,
      .intensity = {100.0f, 100.0f, 100.0f},
      .cos_inner = std::cos(0.25f),
      .cos_outer = std::cos(0.3f),
   };
}

// Éclairement estimé en (0, 0, 0) : flux des photons stockés à moins de radius, par unité d'aire
float irradiance_at_origin(const TestScene& t, int photon_count, float radius, int& stored_count) {
   rt::Pcg32 rng(7, 0);
   double flux = 0.0;
   stored_count = 0;
   for (int i = 0; i < photon_count; ++i) {
      rt::Photon photon;
      if (rt::trace_photon(t.view(), rt::emit_photon(t.view(), rng), rng, photon)) {
         ++stored_count;
         if (rt::length(photon.position) < radius) {
            flux += photon.power.x;
         }
      }
   }
   const auto area = static_cast<double>(rt::kPi * radius * radius);
   return static_cast<float>(flux / (static_cast<double>(photon_count) * area));
}

} // namespace

TEST_CASE("photons : flux émis par une lumière ponctuelle", "[photon]") {
   const rt::PunctualLight point{.position = {0, 0, 1}, .intensity = {2, 3, 4}};
   const TestScene t = make_scene(point, false, false);
   rt::Pcg32 rng(3, 0);
   const rt::PhotonRay photon = rt::emit_photon(t.view(), rng);
   CHECK_THAT(photon.power.y, WithinRel(3.0f * 4.0f * rt::kPi, 1e-5f));

   // Spot : puissance moyenne = I · ∫ atténuation dω, intégrale calculée par quadrature en θ
   const rt::PunctualLight light = spot({0, 0, 2}, {0, 0, -1});
   const TestScene s = make_scene(light, false, false);
   double mean = 0.0;
   constexpr int kCount = 200'000;
   for (int i = 0; i < kCount; ++i) {
      mean += rt::emit_photon(s.view(), rng).power.x / kCount;
   }
   double expected = 0.0;
   constexpr int kSteps = 4096;
   const double theta_max = 0.3;
   for (int i = 0; i < kSteps; ++i) {
      const double theta = (i + 0.5) * theta_max / kSteps;
      const rt::Vec3 d{
         0.0f,
         static_cast<float>(std::sin(theta)),
         -static_cast<float>(std::cos(theta))
      };
      expected += 2.0 * rt::kPi * std::sin(theta) * rt::spot_attenuation(light, d) * theta_max /
                  kSteps;
   }
   CHECK_THAT(mean, WithinRel(100.0 * expected, 0.01));
}

TEST_CASE("photons : sans verre ni miroir, rien n'est stocké (éclairage direct)", "[photon]") {
   const TestScene t = make_scene(spot({0, 0, 2}, {0, 0, -1}), false, false);
   int stored = 0;
   CHECK(irradiance_at_origin(t, 20'000, 0.1f, stored) == 0.0f);
   CHECK(stored == 0);
}

TEST_CASE("photons : à travers une lame d'indice 1, l'éclairement est celui du spot", "[photon]") {
   // Spot à 2 m au-dessus du sol : E = I / d² au centre
   const TestScene t = make_scene(spot({0, 0, 2}, {0, 0, -1}), true, false);
   int stored = 0;
   const float irradiance = irradiance_at_origin(t, 400'000, 0.1f, stored);
   CHECK(stored > 0);
   CHECK_THAT(irradiance, WithinRel(100.0f / 4.0f, 0.03f));
}

TEST_CASE(
   "photons : réfléchis par un miroir, l'éclairement est celui de l'image du spot",
   "[photon]"
) {
   // Spot à 1 m dirigé vers un miroir à 2 m : son image est à 3 m du sol
   const TestScene t = make_scene(spot({0, 0, 1}, {0, 0, 1}), false, true);
   int stored = 0;
   const float irradiance = irradiance_at_origin(t, 400'000, 0.1f, stored);
   CHECK_THAT(irradiance, WithinRel(100.0f / 9.0f, 0.03f));
}
