#include "core/tangent.h"
#include "core/vec3.h"
#include "scene/gltf_loader.h"
#include "scene/tangents.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinAbs;

namespace {

// Carré dans le plan z = 0, normale +z. u croît selon x ; v croît selon -y (origine des textures en
// haut à gauche) ou selon +y si mirrored
rt::Scene make_quad(bool mirrored) {
   rt::Scene s;
   s.positions = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
   s.normals.assign(4, rt::Vec3{0, 0, 1});
   if (mirrored) {
      s.uvs = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
   } else {
      s.uvs = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
   }
   s.indices = {0, 1, 2, 0, 2, 3};
   return s;
}

} // namespace

TEST_CASE(
   "tangentes : selon les u croissants, bitangente vers le haut de la texture",
   "[tangents]"
) {
   rt::Scene s = make_quad(false);
   rt::compute_tangents(s, 0, 4, 6);
   REQUIRE(s.tangents.size() == 4);
   for (const rt::Tangent& t : s.tangents) {
      CHECK_THAT(t.direction.x, WithinAbs(1.0, 1e-6));
      CHECK_THAT(t.direction.y, WithinAbs(0.0, 1e-6));
      // Bitangente = sign × (normale × tangente) = sign × +y : le haut de l'image est vers +y
      CHECK(t.sign == 1.0f);
   }
}

TEST_CASE("tangentes : une texture en miroir inverse la bitangente", "[tangents]") {
   rt::Scene s = make_quad(true);
   rt::compute_tangents(s, 0, 4, 6);
   for (const rt::Tangent& t : s.tangents) {
      CHECK_THAT(t.direction.x, WithinAbs(1.0, 1e-6));
      CHECK(t.sign == -1.0f);
   }
}
