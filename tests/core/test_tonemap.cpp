#include "core/color.h"
#include "core/tonemap.h"
#include "core/vec3.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

using Catch::Matchers::WithinAbs;

TEST_CASE("sRGB : linear_to_srgb est l'inverse de srgb_to_linear", "[tonemap]") {
   for (int i = 0; i <= 100; ++i) {
      const float c = static_cast<float>(i) / 100.0f;
      REQUIRE_THAT(rt::linear_to_srgb(rt::srgb_to_linear(c)), WithinAbs(c, 1e-5));
   }
   // Gris moyen : 18 % de luminance linéaire s'affiche à environ 46 %
   CHECK_THAT(rt::linear_to_srgb(0.18f), WithinAbs(0.4614, 1e-3));
}

TEST_CASE("ACES : noir reste noir, croissant, borné par 1", "[tonemap]") {
   CHECK_THAT(rt::aces_tonemap({0, 0, 0}).x, WithinAbs(0.0, 1e-3));
   float previous = -1.0f;
   for (int i = 0; i < 2000; ++i) {
      const float x = static_cast<float>(i) * 0.05f;
      const float y = rt::aces_tonemap({x, x, x}).x;
      REQUIRE(y >= previous); // un pixel plus lumineux ne devient jamais plus sombre
      REQUIRE(y <= 1.0f);
      previous = y;
   }
   // Les très hautes lumières sont compressées vers le blanc, sans l'atteindre trop tôt
   CHECK(rt::aces_tonemap({1, 1, 1}).x < 0.9f);
   CHECK(rt::aces_tonemap({50, 50, 50}).x > 0.99f);
}

TEST_CASE("ACES : un gris reste gris", "[tonemap]") {
   const rt::Vec3 grey = rt::aces_tonemap({0.4f, 0.4f, 0.4f});
   CHECK_THAT(grey.x, WithinAbs(grey.y, 2e-3));
   CHECK_THAT(grey.y, WithinAbs(grey.z, 2e-3));
}

TEST_CASE("AgX : noir reste noir, croissant, borné par 1, un gris reste gris", "[tonemap]") {
   CHECK_THAT(rt::agx_tonemap({0, 0, 0}).x, WithinAbs(0.0, 1e-3));
   float previous = -1.0f;
   // De 0,001 à 1000, par pas de × 1,25
   for (int i = 0; i < 62; ++i) {
      const float x = 0.001f * std::pow(1.25f, static_cast<float>(i));
      const rt::Vec3 y = rt::agx_tonemap({x, x, x});
      CHECK(y.x >= previous); // constant au-delà d'une radiance de 16
      CHECK(y.x <= 1.0f);
      CHECK_THAT(y.x, WithinAbs(y.y, 2e-3));
      CHECK_THAT(y.y, WithinAbs(y.z, 2e-3));
      previous = y.x;
   }
}

TEST_CASE("AgX : les hautes lumières saturées tendent vers le blanc", "[tonemap]") {
   const auto saturation = [](rt::Vec3 c) { return (c.x - c.z) / c.x; };
   const rt::Vec3 orange{1.0f, 0.3f, 0.05f};
   CHECK(saturation(rt::agx_tonemap(orange * 100.0f)) < saturation(rt::agx_tonemap(orange * 0.5f)));
}

TEST_CASE("affichage : +1 IL d'exposition double la radiance", "[tonemap]") {
   const rt::Vec3 c{0.2f, 0.1f, 0.05f};
   for (const rt::ToneMapping tone_mapping :
        {rt::ToneMapping::Agx, rt::ToneMapping::Aces, rt::ToneMapping::Standard}) {
      const rt::Vec3 a = rt::display_color(c, {.exposure = 1.0f, .tone_mapping = tone_mapping});
      const rt::Vec3 b = rt::display_color(
         c * 2.0f,
         {.exposure = 0.0f, .tone_mapping = tone_mapping}
      );
      CHECK_THAT(a.x, WithinAbs(b.x, 1e-5));
      CHECK_THAT(a.z, WithinAbs(b.z, 1e-5));
   }
}
