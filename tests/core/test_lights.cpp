#include "core/bsdf.h"
#include "core/environment.h"
#include "core/light.h"
#include "core/path_tracer.h"
#include "core/ray.h"
#include "core/rng.h"
#include "core/scene_view.h"
#include "core/vec2.h"
#include "core/vec3.h"
#include "scene/bvh_builder.h"
#include "scene/gltf_loader.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>
#include <vector>

// Une lumière ponctuelle au-dessus d'un sol mat, dans un environnement noir. Le rayon caméra touche
// le sol ; l'éclairage direct vaut exactement f · E · cos θ (E : éclairement reçu), et le rebond
// suivant part vers le ciel noir. Chaque échantillon vaut donc exactement la valeur attendue.

namespace {

constexpr rt::SurfaceParams kFloor{
   .base_color = {0.8f, 0.8f, 0.8f},
   .metallic = 0.0f,
   .roughness = 1.0f
};
constexpr rt::Vec3 kHitPoint{0.5f, 0.0f, 0.0f};

struct TestScene {
   rt::Scene scene;
   rt::Bvh bvh;

   [[nodiscard]] rt::SceneView view() const {
      return {
         .nodes = bvh.nodes.data(),
         .triangle_indices = bvh.triangle_indices.data(),
         .positions = scene.positions.data(),
         .indices = scene.indices.data(),
         .materials = scene.materials.data(),
         .material_indices = scene.material_indices.data(),
         .emissive_triangles = nullptr,
         .emissive_triangle_count = 0,
         .punctual_lights = scene.punctual_lights.data(),
         .punctual_light_count = static_cast<std::uint32_t>(scene.punctual_lights.size()),
         .uvs = scene.uvs.data(),
         .environment = scene.environment,
         .caustics = {},
      };
   }
};

// Sol carré z = 0 de 20 × 20, normale +z, plus éventuellement un petit carré occultant à z = 1
TestScene make_scene(const std::vector<rt::PunctualLight>& lights, bool occluder) {
   TestScene t;
   rt::Scene& s = t.scene;
   const auto add_quad = [&](float half, float z) {
      const auto base = static_cast<std::uint32_t>(s.positions.size());
      s.positions.insert(
         s.positions.end(),
         {{-half, -half, z}, {half, -half, z}, {half, half, z}, {-half, half, z}}
      );
      s.uvs.insert(s.uvs.end(), 4, rt::Vec2{0, 0});
      s.indices.insert(s.indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
      s.material_indices.insert(s.material_indices.end(), 2, 0u);
   };
   add_quad(10.0f, 0.0f);
   if (occluder) {
      add_quad(1.0f, 1.0f);
   }
   s.materials = {{
      .albedo = kFloor.base_color,
      .emission = {0, 0, 0},
      .metallic = kFloor.metallic,
      .roughness = kFloor.roughness,
   }};
   s.punctual_lights = lights;
   s.environment = {.bottom = {0, 0, 0}, .top = {0, 0, 0}};
   t.bvh = rt::build_bvh(s);
   return t;
}

// Lumière renvoyée vers le haut par le sol au point touché, depuis un rayon vertical
rt::Vec3 render_hit_point(const TestScene& t, std::uint32_t& ray_count) {
   const rt::Ray ray{kHitPoint + rt::Vec3{0.0f, 0.0f, 0.5f}, {0.0f, 0.0f, -1.0f}};
   rt::Pcg32 rng(1, 0);
   return rt::trace_path(ray, t.view(), rng, ray_count);
}

// Valeur attendue : f(wo = verticale, wi) · E · cos θ
rt::Vec3 expected_radiance(rt::Vec3 wi, rt::Vec3 irradiance) {
   return rt::principled::eval(kFloor, {0, 0, 1}, wi) * irradiance * wi.z;
}

void check_close(rt::Vec3 actual, rt::Vec3 expected) {
   INFO("obtenu " << actual.x << ", attendu " << expected.x);
   CHECK_THAT(actual.x, Catch::Matchers::WithinRel(expected.x, 1e-4f));
   CHECK_THAT(actual.y, Catch::Matchers::WithinRel(expected.y, 1e-4f));
   CHECK_THAT(actual.z, Catch::Matchers::WithinRel(expected.z, 1e-4f));
}

} // namespace

TEST_CASE("lumière ponctuelle : éclairement I cos θ / d²", "[lights]") {
   const rt::PunctualLight light{
      .type = rt::PunctualLight::Type::Point,
      .position = {0.0f, 0.0f, 2.0f},
      .intensity = {4.0f, 2.0f, 1.0f},
   };
   const TestScene t = make_scene({light}, false);
   std::uint32_t ray_count = 0;
   const rt::Vec3 radiance = render_hit_point(t, ray_count);

   const rt::Vec3 to_light = light.position - kHitPoint;
   const float d2 = rt::length_squared(to_light);
   check_close(radiance, expected_radiance(rt::normalize(to_light), light.intensity / d2));
   // Rayon caméra, rayon d'ombre, puis rebond vers le ciel (sauf si le tirage de la BSDF échoue)
   CHECK(ray_count >= 2u);
   CHECK(ray_count <= 3u);
}

TEST_CASE("spot : atténuation entre les cônes intérieur et extérieur", "[lights]") {
   // Le point touché est vu sous atan(0,5 / 2) ≈ 14° depuis le spot, qui éclaire vers le bas
   const float cos_inner = std::cos(10.0f * rt::kPi / 180.0f);
   const float cos_outer = std::cos(20.0f * rt::kPi / 180.0f);
   const rt::PunctualLight spot{
      .type = rt::PunctualLight::Type::Spot,
      .position = {0.0f, 0.0f, 2.0f},
      .direction = {0.0f, 0.0f, -1.0f},
      .intensity = {4.0f, 4.0f, 4.0f},
      .cos_inner = cos_inner,
      .cos_outer = cos_outer,
   };
   std::uint32_t ray_count = 0;
   const rt::Vec3 radiance = render_hit_point(make_scene({spot}, false), ray_count);

   const rt::Vec3 to_light = spot.position - kHitPoint;
   const float d2 = rt::length_squared(to_light);
   const rt::Vec3 wi = rt::normalize(to_light);
   const float t = (rt::dot(spot.direction, -wi) - cos_outer) / (cos_inner - cos_outer);
   REQUIRE(t > 0.0f);
   REQUIRE(t < 1.0f);
   check_close(radiance, expected_radiance(wi, spot.intensity * (t * t / d2)));

   // Hors du cône extérieur : aucune lumière
   rt::PunctualLight narrow = spot;
   narrow.cos_inner = std::cos(4.0f * rt::kPi / 180.0f);
   narrow.cos_outer = std::cos(8.0f * rt::kPi / 180.0f);
   CHECK(render_hit_point(make_scene({narrow}, false), ray_count) == rt::Vec3{0, 0, 0});
}

TEST_CASE("lumière directionnelle : éclairement constant", "[lights]") {
   const rt::Vec3 direction = rt::normalize({1.0f, 0.0f, -2.0f});
   const rt::PunctualLight sun{
      .type = rt::PunctualLight::Type::Directional,
      .direction = direction,
      .intensity = {3.0f, 3.0f, 3.0f},
   };
   std::uint32_t ray_count = 0;
   const rt::Vec3 radiance = render_hit_point(make_scene({sun}, false), ray_count);
   check_close(radiance, expected_radiance(-direction, sun.intensity));
}

TEST_CASE("lumières ponctuelles : ombre portée et somme de plusieurs lumières", "[lights]") {
   const rt::PunctualLight a{.position = {0.0f, 0.0f, 2.0f}, .intensity = {4.0f, 4.0f, 4.0f}};
   const rt::PunctualLight b{.position = {3.0f, 0.0f, 2.0f}, .intensity = {1.0f, 1.0f, 1.0f}};

   // Un carré entre le point et la lumière a : plus rien ne passe
   std::uint32_t ray_count = 0;
   CHECK(render_hit_point(make_scene({a}, true), ray_count) == rt::Vec3{0, 0, 0});

   // Deux lumières : une seule est tirée par échantillon, mais la moyenne vaut la somme
   const TestScene t = make_scene({a, b}, false);
   rt::Vec3 expected{0, 0, 0};
   for (const rt::PunctualLight& light : {a, b}) {
      const rt::Vec3 to_light = light.position - kHitPoint;
      expected += expected_radiance(
         rt::normalize(to_light),
         light.intensity / rt::length_squared(to_light)
      );
   }
   constexpr int kSamples = 20'000;
   const rt::Ray ray{kHitPoint + rt::Vec3{0.0f, 0.0f, 0.5f}, {0.0f, 0.0f, -1.0f}};
   double sum = 0.0;
   for (int i = 0; i < kSamples; ++i) {
      rt::Pcg32 rng(2, static_cast<std::uint64_t>(i));
      sum += rt::trace_path(ray, t.view(), rng, ray_count).x;
   }
   CHECK_THAT(sum / kSamples, Catch::Matchers::WithinRel(static_cast<double>(expected.x), 0.02));
}

TEST_CASE("écrêtage des fireflies : l'éclairage direct n'est jamais écrêté", "[lights][clamp]") {
   // Seuil minuscule : l'éclairage direct reste exact
   const rt::PunctualLight light{
      .position = {0.0f, 0.0f, 2.0f},
      .intensity = {40.0f, 40.0f, 40.0f}
   };
   const TestScene t = make_scene({light}, false);
   rt::SceneView view = t.view();
   view.firefly_clamp = 1e-3f;
   const rt::Ray ray{kHitPoint + rt::Vec3{0.0f, 0.0f, 0.5f}, {0.0f, 0.0f, -1.0f}};
   rt::Pcg32 rng(1, 0);
   std::uint32_t ray_count = 0;
   const rt::Vec3 radiance = rt::trace_path(ray, view, rng, ray_count);

   const rt::Vec3 to_light = light.position - kHitPoint;
   const rt::Vec3 expected = expected_radiance(
      rt::normalize(to_light),
      light.intensity / rt::length_squared(to_light)
   );
   REQUIRE(rt::max_component(expected) > 1e-3f);
   check_close(radiance, expected);
}
