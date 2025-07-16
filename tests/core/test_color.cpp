#include "core/color.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinAbs;

TEST_CASE("srgb_to_linear : valeurs de référence", "[color]") {
   CHECK_THAT(rt::srgb_to_linear(0.0f), WithinAbs(0.0, 1e-7));
   CHECK_THAT(rt::srgb_to_linear(1.0f), WithinAbs(1.0, 1e-6));
   // Gris moyen sRGB 128/255 ≈ 21,6 % de luminance linéaire
   CHECK_THAT(rt::srgb_to_linear(128.0f / 255.0f), WithinAbs(0.2158605, 1e-5));
}

TEST_CASE("srgb_to_linear : continue au raccord des deux formules", "[color]") {
   const float below = rt::srgb_to_linear(0.04045f);
   const float above = rt::srgb_to_linear(0.04046f);
   CHECK_THAT(below, WithinAbs(0.0031308, 1e-6));
   CHECK(above > below);
   CHECK_THAT(above - below, WithinAbs(0.0, 1e-5));
}
