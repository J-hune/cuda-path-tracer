#include "core/bsdf.h"
#include "core/light.h"
#include "core/path_tracer.h"
#include "core/photon_map.h"
#include "core/photon_tracer.h"
#include "core/ray.h"
#include "core/rng.h"
#include "core/scene_view.h"
#include "core/vec2.h"
#include "core/vec3.h"
#include "cpu_photon_map.h"
#include "scene/bvh_builder.h"
#include "scene/gltf_loader.h"
#include "scene/photon_targets.h"

#include <catch2/catch_message.hpp>
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
constexpr std::uint32_t kLightMaterial = 3;
constexpr rt::SurfaceParams kFloor{.base_color = {0.8f, 0.8f, 0.8f}};

struct TestScene {
   rt::Scene scene;
   rt::Bvh bvh;
   std::vector<rt::PhotonTarget> targets; // vide : émission dans toutes les directions

   [[nodiscard]] rt::SceneView view() const {
      rt::SceneView view;
      view.nodes = bvh.nodes.data();
      view.triangle_indices = bvh.triangle_indices.data();
      view.bvh_triangles = bvh.triangles.data();
      view.positions = scene.positions.data();
      view.indices = scene.indices.data();
      view.materials = scene.materials.data();
      view.material_indices = scene.material_indices.data();
      view.emissive_triangles = scene.emissive_triangles.data();
      view.emissive_triangle_count = static_cast<std::uint32_t>(scene.emissive_triangles.size());
      view.punctual_lights = scene.punctual_lights.data();
      view.punctual_light_count = static_cast<std::uint32_t>(scene.punctual_lights.size());
      view.uvs = scene.uvs.data();
      view.photon_targets = targets.data();
      view.photon_target_count = static_cast<std::uint32_t>(targets.size());
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
      {.albedo = {0, 0, 0}, .emission = {5, 5, 5}},
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

// Sol mat, lame de verre d'indice 1, et carré émissif au-dessus de la lame
// La lampe regarde vers le bas (vers le sol) sauf si light_up ; double_sided : elle éclaire des
// deux côtés
TestScene make_area_light_scene(bool light_up = false, bool double_sided = true) {
   TestScene t;
   rt::Scene& s = t.scene;
   s.materials = make_scene({}, false, false).scene.materials;
   s.materials[kLightMaterial].double_sided = double_sided;
   add_quad(s, 10.0f, 0.0f, true, kFloorMaterial);
   add_quad(s, 2.0f, 1.1f, true, kGlassMaterial);
   add_quad(s, 2.0f, 1.0f, false, kGlassMaterial);
   add_quad(s, 0.5f, 2.0f, light_up, kLightMaterial);
   s.emissive_triangles = {6, 7};
   t.bvh = rt::build_bvh(s);
   return t;
}

// Moyenne de trace_path pour un rayon vertical qui touche le sol en (0, 0, 0), sous la lame
rt::Vec3 floor_radiance(const rt::SceneView& view, int samples) {
   const rt::Ray ray{{0.0f, 0.0f, 0.5f}, {0.0f, 0.0f, -1.0f}};
   rt::Vec3 sum{0.0f, 0.0f, 0.0f};
   std::uint32_t ray_count = 0;
   for (int i = 0; i < samples; ++i) {
      rt::Pcg32 rng = rt::Pcg32::make_pixel_rng(0, static_cast<std::uint32_t>(i));
      sum += rt::trace_path(ray, view, rng, ray_count);
   }
   return sum / static_cast<float>(samples);
}

TestScene targeted(TestScene t) {
   t.targets = rt::photon_targets(t.scene);
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
   CHECK_THAT(irradiance_at_origin(t, 400'000, 0.1f, stored), WithinRel(100.0f / 4.0f, 0.03f));
   CHECK_THAT(
      irradiance_at_origin(targeted(t), 400'000, 0.1f, stored),
      WithinRel(100.0f / 4.0f, 0.03f)
   );
}

TEST_CASE("photons : une lame dispersive d'indice 1 garde l'éclairement blanc", "[photon]") {
   // Indice 1 à toutes les longueurs d'onde : seule la couleur tirée change, en moyenne blanche
   TestScene t = make_scene(spot({0, 0, 2}, {0, 0, -1}), true, false);
   t.scene.materials[kGlassMaterial].dispersion = 1.0f;
   const rt::SceneView view = t.view();
   rt::Pcg32 rng(7, 0);
   rt::Vec3 flux{0.0f, 0.0f, 0.0f};
   constexpr int kCount = 400'000;
   for (int i = 0; i < kCount; ++i) {
      rt::Photon photon;
      if (rt::trace_photon(view, rt::emit_photon(view, rng), rng, photon) &&
          rt::length(photon.position) < 0.1f) {
         flux += photon.power;
      }
   }
   const rt::Vec3 irradiance = flux / (static_cast<float>(kCount) * rt::kPi * 0.01f);
   CHECK_THAT(irradiance.x, WithinRel(100.0f / 4.0f, 0.04f));
   CHECK_THAT(irradiance.y, WithinRel(100.0f / 4.0f, 0.04f));
   CHECK_THAT(irradiance.z, WithinRel(100.0f / 4.0f, 0.04f));
}

TEST_CASE("photons : l'émission dirigée vers une petite lame garde l'éclairement", "[photon]") {
   // Lampe ponctuelle à 3 m au-dessus d'une lame de 60 cm : E = I / 3² sous la lame
   TestScene t = make_scene({.position = {0, 0, 3}, .intensity = {100, 100, 100}}, false, false);
   add_quad(t.scene, 0.3f, 1.1f, true, kGlassMaterial);
   add_quad(t.scene, 0.3f, 1.0f, false, kGlassMaterial);
   t.bvh = rt::build_bvh(t.scene);

   int stored = 0;
   int stored_targeted = 0;
   const float uniform = irradiance_at_origin(t, 400'000, 0.1f, stored);
   const float irradiance = irradiance_at_origin(targeted(t), 400'000, 0.1f, stored_targeted);
   CHECK_THAT(irradiance, WithinRel(100.0f / 9.0f, 0.03f));
   // Presque tous les photons dirigés traversent la lame, contre 1 % sans émission dirigée
   CHECK(stored_targeted > 20 * stored);
   CHECK(uniform > 0.0f);
}

TEST_CASE(
   "photons : réfléchis par un miroir, l'éclairement est celui de l'image du spot",
   "[photon]"
) {
   // Spot à 1 m dirigé vers un miroir à 2 m : son image est à 3 m du sol
   const TestScene t = make_scene(spot({0, 0, 1}, {0, 0, 1}), false, true);
   for (const TestScene& scene : {t, targeted(t)}) {
      INFO("cibles : " << scene.targets.size());
      int stored = 0;
      const float irradiance = irradiance_at_origin(scene, 2'000'000, 0.1f, stored);
      CHECK_THAT(irradiance, WithinRel(100.0f / 9.0f, 0.03f));
   }
}

TEST_CASE("caustiques : sous la lame, les photons apportent la lumière du spot", "[photon]") {
   // Le path tracer seul ne voit rien : l'ombre portée par le verre bloque l'éclairage direct
   const TestScene t = make_scene(spot({0, 0, 2}, {0, 0, -1}), true, false);
   CHECK(floor_radiance(t.view(), 16).x == 0.0f);

   const rt::test::CpuPhotonMap map(rt::test::trace_photons(t.view(), 400'000, 9), 1u << 16, 0.1f);
   rt::SceneView view = t.view();
   view.caustics = map.view;
   const float expected = rt::principled::eval(kFloor, {0, 0, 1}, {0, 0, 1}).x * (100.0f / 4.0f);
   CHECK_THAT(floor_radiance(view, 16).x, WithinRel(expected, 0.03f));
}

TEST_CASE("caustiques : photons et path tracer ne comptent pas deux fois la lumière", "[photon]") {
   // Lampe surfacique : le path tracer seul trouve aussi la caustique, en touchant la lampe à
   // travers le verre. Avec les photons, ces chemins sont exclus : la moyenne reste la même.
   const TestScene t = make_area_light_scene();
   const rt::Vec3 path_traced = floor_radiance(t.view(), 400'000);

   const rt::test::CpuPhotonMap map(
      rt::test::trace_photons(t.view(), 4'000'000, 9),
      1u << 16,
      0.15f
   );
   rt::SceneView view = t.view();
   view.caustics = map.view;
   const rt::Vec3 with_photons = floor_radiance(view, 64);

   CHECK(path_traced.x > 0.0f);
   CHECK_THAT(with_photons.x, WithinRel(path_traced.x, 0.03f));

   // Émission dirigée depuis la lampe surfacique vers la lame
   const TestScene targeted_scene = targeted(t);
   const rt::test::CpuPhotonMap targeted_map(
      rt::test::trace_photons(targeted_scene.view(), 4'000'000, 9),
      1u << 16,
      0.15f
   );
   rt::SceneView targeted_view = targeted_scene.view();
   targeted_view.caustics = targeted_map.view;
   CHECK_THAT(floor_radiance(targeted_view, 64).x, WithinRel(path_traced.x, 0.03f));
}

TEST_CASE("lampe à simple face : n'éclaire que du côté de sa normale", "[photon]") {
   // Tournée vers le sol, elle donne le même éclairage qu'une lampe à double face
   const TestScene both = make_area_light_scene(false, true);
   const TestScene front = make_area_light_scene(false, false);
   const rt::Vec3 reference = floor_radiance(both.view(), 100'000);
   CHECK_THAT(floor_radiance(front.view(), 100'000).x, WithinRel(reference.x, 0.03f));

   // Les photons d'une lampe à simple face partent tous vers le sol : même éclairement
   const rt::test::CpuPhotonMap map(
      rt::test::trace_photons(front.view(), 2'000'000, 9),
      1u << 16,
      0.15f
   );
   rt::SceneView view = front.view();
   view.caustics = map.view;
   CHECK_THAT(floor_radiance(view, 64).x, WithinRel(reference.x, 0.03f));

   // Tournée vers le plafond, elle n'éclaire plus le sol, ni directement ni par les photons
   const TestScene away = make_area_light_scene(true, false);
   CHECK(floor_radiance(away.view(), 10'000).x == 0.0f);
   CHECK(rt::test::trace_photons(away.view(), 100'000, 9).empty());
}
