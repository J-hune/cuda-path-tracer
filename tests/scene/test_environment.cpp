#include "core/environment.h"
#include "core/rng.h"
#include "core/vec2.h"
#include "core/vec3.h"
#include "scene/environment_map.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>
#include <numbers>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

// HDRI 32 × 16 aux luminances très inégales, pour que le tirage soit loin de l'uniforme
rt::EnvironmentImage make_image(float rotation) {
   rt::EnvironmentImage image;
   image.width = 32;
   image.height = 16;
   image.rotation = rotation;
   rt::Pcg32 rng(11, 0);
   for (std::uint32_t i = 0; i < image.width * image.height; ++i) {
      const float l = std::pow(rng.next_float(), 4.0f) * 10.0f;
      image.texels.push_back({l, 0.5f * l, 0.25f * l});
   }
   image.texels[37] = {500.0f, 400.0f, 300.0f}; // un « soleil »
   rt::build_environment_distribution(image);
   return image;
}

} // namespace

TEST_CASE("HDRI : projection de Blender, aller-retour avec rotation", "[environment]") {
   const rt::EnvironmentImage image = make_image(0.0f);
   const rt::EnvironmentMap map = image.view();
   // Comme Cycles : le centre de l'image regarde vers +x, le haut de l'image vers +y
   const rt::Vec2 center = map.direction_to_uv({1.0f, 0.0f, 0.0f});
   CHECK_THAT(center.x, WithinAbs(0.5, 1e-6));
   CHECK_THAT(center.y, WithinAbs(0.5, 1e-6));
   CHECK_THAT(map.direction_to_uv({0.0f, 1.0f, 0.0f}).y, WithinAbs(0.0, 1e-6));

   for (const float rotation : {0.0f, 0.7f, -2.0f}) {
      const rt::EnvironmentMap rotated = make_image(rotation).view();
      for (const rt::Vec3 d : {rt::Vec3{0.3f, 0.5f, -0.8f}, rt::Vec3{-0.9f, -0.2f, 0.1f}}) {
         const rt::Vec3 back = rotated.uv_to_direction(rotated.direction_to_uv(rt::normalize(d)));
         CHECK(rt::length(back - rt::normalize(d)) < 1e-5f);
      }
   }
}

TEST_CASE("HDRI : la rotation suit le sens de Blender", "[environment]") {
   // Rotation de 90° (Mapping de Blender, autour de Z) : le centre de l'image, +x sans rotation,
   // se retrouve vers -y Blender, c'est-à-dire +z en glTF
   const rt::EnvironmentMap rotated = make_image(std::numbers::pi_v<float> / 2.0f).view();
   const rt::Vec2 center = rotated.direction_to_uv({0.0f, 0.0f, 1.0f});
   CHECK_THAT(center.x, WithinAbs(0.5, 1e-6));
}

TEST_CASE("HDRI : le tirage suit sa densité, qui s'intègre à 1 sur la sphère", "[environment]") {
   const rt::EnvironmentImage image = make_image(0.4f);
   const rt::EnvironmentMap map = image.view();

   // Intégrale de la densité sur la sphère, par quadrature en (θ, φ)
   constexpr int kSteps = 400;
   double integral = 0.0;
   for (int i = 0; i < kSteps; ++i) {
      const double theta = (i + 0.5) * std::numbers::pi / kSteps;
      for (int j = 0; j < 2 * kSteps; ++j) {
         const double phi = (j + 0.5) * std::numbers::pi / kSteps;
         const rt::Vec3 d{
            static_cast<float>(std::sin(theta) * std::cos(phi)),
            static_cast<float>(std::cos(theta)),
            static_cast<float>(std::sin(theta) * std::sin(phi)),
         };
         integral += map.pdf(d) * std::sin(theta) * (std::numbers::pi / kSteps) *
                     (std::numbers::pi / kSteps);
      }
   }
   CHECK_THAT(integral, WithinRel(1.0, 0.01));

   // Estimation de ∫ L dω par le tirage, comparée à la somme exacte sur les texels
   double exact = 0.0;
   for (std::uint32_t row = 0; row < image.height; ++row) {
      const double theta0 = std::numbers::pi * row / image.height;
      const double theta1 = std::numbers::pi * (row + 1) / image.height;
      const double solid_angle = (std::cos(theta0) - std::cos(theta1)) * 2.0 * std::numbers::pi /
                                 image.width;
      for (std::uint32_t column = 0; column < image.width; ++column) {
         exact += image.texels[(row * image.width) + column].x * solid_angle;
      }
   }
   rt::Pcg32 rng(5, 0);
   double estimate = 0.0;
   int mismatches = 0;
   constexpr int kSamples = 200'000;
   for (int i = 0; i < kSamples; ++i) {
      const rt::EnvironmentSample s = map.sample({rng.next_float(), rng.next_float()});
      REQUIRE(s.pdf > 0.0f);
      // Une direction tirée sur la frontière de deux texels peut retomber dans le voisin
      mismatches += std::fabs(s.pdf - map.pdf(s.direction)) > 1e-3f * s.pdf ? 1 : 0;
      estimate += map.radiance(s.direction).x / s.pdf / kSamples;
   }
   CHECK(mismatches < kSamples / 1000);
   CHECK_THAT(estimate, WithinRel(exact, 0.01));
}
