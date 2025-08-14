#include "core/material.h"
#include "core/path_tracer.h"
#include "core/scene_view.h"
#include "core/tangent.h"
#include "core/texture.h"
#include "core/vec3.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>

using Catch::Matchers::WithinAbs;

namespace {

constexpr std::array<std::uint32_t, 3> kIndices{0, 1, 2};
constexpr rt::Vec3 kGeometric{0.0f, 0.0f, 1.0f};

rt::SceneView view_with(const std::array<rt::Vec3, 3>& normals) {
   rt::SceneView view;
   view.indices = kIndices.data();
   view.normals = normals.data();
   return view;
}

void check_vec(rt::Vec3 actual, rt::Vec3 expected) {
   CHECK_THAT(actual.x, WithinAbs(expected.x, 1e-5));
   CHECK_THAT(actual.y, WithinAbs(expected.y, 1e-5));
   CHECK_THAT(actual.z, WithinAbs(expected.z, 1e-5));
}

} // namespace

TEST_CASE("normale d'ombrage : interpolée entre les sommets", "[shading_normal]") {
   const std::array<rt::Vec3, 3> normals{
      rt::normalize({-1, 0, 1}),
      rt::normalize({1, 0, 1}),
      rt::Vec3{0, 0, 1}
   };
   const rt::SceneView view = view_with(normals);
   // Aux sommets : la normale du sommet
   check_vec(
      rt::shading_normal(view, {.t = 1, .triangle = 0, .u = 0, .v = 0}, kGeometric),
      normals[0]
   );
   check_vec(
      rt::shading_normal(view, {.t = 1, .triangle = 0, .u = 1, .v = 0}, kGeometric),
      normals[1]
   );
   // Au milieu de l'arête v0-v1 : les deux normales symétriques se compensent
   check_vec(
      rt::shading_normal(view, {.t = 1, .triangle = 0, .u = 0.5f, .v = 0}, kGeometric),
      {0, 0, 1}
   );
}

TEST_CASE(
   "normale d'ombrage : du côté de la normale géométrique, sinon géométrique",
   "[shading_normal]"
) {
   const std::array<rt::Vec3, 3> flipped{
      rt::Vec3{0, 0, -1},
      rt::Vec3{0, 0, -1},
      rt::Vec3{0, 0, -1}
   };
   check_vec(
      rt::shading_normal(
         view_with(flipped),
         {.t = 1, .triangle = 0, .u = 0.2f, .v = 0.3f},
         kGeometric
      ),
      {0, 0, 1}
   );

   // Ombrage plat (normales nulles) et scène sans normales : la normale géométrique
   const std::array<rt::Vec3, 3> flat{rt::Vec3{0, 0, 0}, rt::Vec3{0, 0, 0}, rt::Vec3{0, 0, 0}};
   check_vec(
      rt::shading_normal(
         view_with(flat),
         {.t = 1, .triangle = 0, .u = 0.2f, .v = 0.3f},
         kGeometric
      ),
      kGeometric
   );
   check_vec(
      rt::shading_normal(
         rt::SceneView{},
         {.t = 1, .triangle = 0, .u = 0.2f, .v = 0.3f},
         kGeometric
      ),
      kGeometric
   );
}

TEST_CASE(
   "terminateur d'ombre : neutre si les normales coïncident, borné sinon",
   "[shading_normal]"
) {
   const rt::Vec3 wi = rt::normalize({0.3f, 0.2f, 0.5f});
   CHECK_THAT(rt::shadow_terminator(kGeometric, kGeometric, wi), WithinAbs(1.0, 1e-5));

   // Normale d'ombrage inclinée de 30° : atténuation dans [0, 1], plus forte en lumière rasante
   const rt::Vec3 tilted = rt::normalize({std::sin(0.52f), 0.0f, std::cos(0.52f)});
   const float grazing = rt::shadow_terminator(
      kGeometric,
      tilted,
      rt::normalize({1.0f, 0.0f, 0.05f})
   );
   const float overhead = rt::shadow_terminator(
      kGeometric,
      tilted,
      rt::normalize({0.3f, 0.0f, 1.0f})
   );
   CHECK(grazing >= 0.0f);
   CHECK(grazing < overhead);
   CHECK(overhead <= 1.0f);
   // Lumière sous la normale d'ombrage : rien
   CHECK(rt::shadow_terminator(kGeometric, tilted, rt::normalize({-1.0f, 0.0f, 0.1f})) == 0.0f);
}

TEST_CASE(
   "normal mapping : texture neutre, inclinaison vers la tangente, intensité",
   "[shading_normal]"
) {
   const std::array<rt::Vec3, 3> normals{rt::Vec3{0, 0, 1}, rt::Vec3{0, 0, 1}, rt::Vec3{0, 0, 1}};
   const std::array<rt::Tangent, 3> tangents{
      rt::Tangent{.direction = {1, 0, 0}, .sign = 1.0f},
      rt::Tangent{.direction = {1, 0, 0}, .sign = 1.0f},
      rt::Tangent{.direction = {1, 0, 0}, .sign = 1.0f},
   };
   // Texture 2 × 1 : texel neutre (0,5 ; 0,5 ; 1), puis incliné à 45° vers la tangente
   const std::array<rt::Texture, 1> textures{rt::Texture{.offset = 0, .width = 2, .height = 1}};
   const float tilt = 0.5f + (0.5f * std::sqrt(0.5f));
   const std::array<rt::Vec3, 2> texels{rt::Vec3{0.5f, 0.5f, 1.0f}, rt::Vec3{tilt, 0.5f, tilt}};
   rt::SceneView view = view_with(normals);
   view.tangents = tangents.data();
   view.textures = textures.data();
   view.texels = texels.data();
   const rt::Hit hit{.t = 1, .triangle = 0, .u = 0.2f, .v = 0.3f};
   rt::Material material{.albedo = {1, 1, 1}, .emission = {0, 0, 0}, .normal_texture = 0};

   check_vec(rt::mapped_normal(view, hit, material, {0.25f, 0.5f}, kGeometric), {0, 0, 1});
   check_vec(
      rt::mapped_normal(view, hit, material, {0.75f, 0.5f}, kGeometric),
      rt::normalize({1, 0, 1})
   );
   // Intensité nulle : seule la composante normale reste
   material.normal_scale = 0.0f;
   check_vec(rt::mapped_normal(view, hit, material, {0.75f, 0.5f}, kGeometric), {0, 0, 1});
   // Vue de l'autre côté : la normale perturbée est retournée avec la surface
   material.normal_scale = 1.0f;
   check_vec(
      rt::mapped_normal(view, hit, material, {0.75f, 0.5f}, -kGeometric),
      rt::normalize({-1, 0, -1})
   );
}
