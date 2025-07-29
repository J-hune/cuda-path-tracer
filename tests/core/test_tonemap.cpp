#include "core/color.h"
#include "core/tonemap.h"
#include "core/vec3.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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
