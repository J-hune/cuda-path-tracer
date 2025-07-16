#include "core/texture.h"
#include "core/vec3.h"

#include <array>
#include <catch2/catch_test_macros.hpp>

namespace {
// Deux textures 2 × 2 à la suite ; la seconde commence à l'offset 4
// Seconde texture : ligne 0 = rouge, vert ; ligne 1 = bleu, blanc
constexpr std::array<rt::Vec3, 8> kTexels{{
   {0, 0, 0},
   {0, 0, 0},
   {0, 0, 0},
   {0, 0, 0},
   {1, 0, 0},
   {0, 1, 0},
   {0, 0, 1},
   {1, 1, 1},
}};
constexpr rt::Texture kTexture{.offset = 4, .width = 2, .height = 2};
} // namespace

TEST_CASE("sample_texture : chaque quart de l'image", "[texture]") {
   CHECK(rt::sample_texture(kTexture, kTexels.data(), {0.25f, 0.25f}) == rt::Vec3{1, 0, 0});
   CHECK(rt::sample_texture(kTexture, kTexels.data(), {0.75f, 0.25f}) == rt::Vec3{0, 1, 0});
   CHECK(rt::sample_texture(kTexture, kTexels.data(), {0.25f, 0.75f}) == rt::Vec3{0, 0, 1});
   CHECK(rt::sample_texture(kTexture, kTexels.data(), {0.75f, 0.75f}) == rt::Vec3{1, 1, 1});
}

TEST_CASE("sample_texture : bords et répétition", "[texture]") {
   // u = v = 1 reste dans l'image (dernier texel), sans lire au-delà
   CHECK(rt::sample_texture(kTexture, kTexels.data(), {1.0f, 1.0f}) == rt::Vec3{1, 0, 0});
   CHECK(rt::sample_texture(kTexture, kTexels.data(), {0.999f, 0.999f}) == rt::Vec3{1, 1, 1});
   // La texture se répète : u = 1,25 équivaut à 0,25, u = -0,25 à 0,75
   CHECK(rt::sample_texture(kTexture, kTexels.data(), {1.25f, 0.25f}) == rt::Vec3{1, 0, 0});
   CHECK(rt::sample_texture(kTexture, kTexels.data(), {-0.25f, 0.25f}) == rt::Vec3{0, 1, 0});
}
