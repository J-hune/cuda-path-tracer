#include "core/spectrum.h"
#include "core/vec3.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("spectre : couleurs positives, une lumière blanche reste blanche", "[spectrum]") {
   // Moyenne sur les longueurs d'onde tirées uniformément : (1, 1, 1)
   constexpr int kSteps = 100'000;
   rt::Vec3 mean{0.0f, 0.0f, 0.0f};
   for (int i = 0; i < kSteps; ++i) {
      const float u = (static_cast<float>(i) + 0.5f) / static_cast<float>(kSteps);
      const rt::Vec3 c = rt::wavelength_rgb(rt::sample_wavelength(u));
      CHECK(rt::min_component(c) >= 0.0f);
      mean += c / static_cast<float>(kSteps);
   }
   CHECK_THAT(mean.x, WithinRel(1.0f, 1e-3f));
   CHECK_THAT(mean.y, WithinRel(1.0f, 1e-3f));
   CHECK_THAT(mean.z, WithinRel(1.0f, 1e-3f));
}

TEST_CASE("spectre : le bleu, le vert et le rouge aux bonnes longueurs d'onde", "[spectrum]") {
   const rt::Vec3 blue = rt::wavelength_rgb(450.0f);
   const rt::Vec3 green = rt::wavelength_rgb(530.0f);
   const rt::Vec3 red = rt::wavelength_rgb(630.0f);
   CHECK(blue.z > blue.x);
   CHECK(blue.z > blue.y);
   CHECK(green.y > green.x);
   CHECK(green.y > green.z);
   CHECK(red.x > red.y);
   CHECK(red.x > red.z);
}

TEST_CASE("dispersion : indice d'Abbe retrouvé sur les raies F, d et C", "[spectrum]") {
   // Nombre d'Abbe V = (n_d - 1) / (n_F - n_C), avec dispersion = 20 / V
   constexpr float kIor = 1.6f;
   constexpr float kAbbe = 12.0f;
   const float n_d = rt::dispersive_ior(kIor, 20.0f / kAbbe, 587.56f);
   const float n_f = rt::dispersive_ior(kIor, 20.0f / kAbbe, 486.13f);
   const float n_c = rt::dispersive_ior(kIor, 20.0f / kAbbe, 656.27f);
   CHECK_THAT(n_d, WithinAbs(kIor, 1e-4));
   CHECK_THAT((n_d - 1.0f) / (n_f - n_c), WithinRel(kAbbe, 1e-3f));
   // Sans dispersion : indice constant
   CHECK(rt::dispersive_ior(kIor, 0.0f, 450.0f) == kIor);
}
