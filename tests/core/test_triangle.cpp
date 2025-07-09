#include "core/ray.h"
#include "core/triangle.h"
#include "core/vec3.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("intersection d'un triangle", "[triangle]") {
   const rt::Vec3 v0{0.0f, 0.0f, 0.0f};
   const rt::Vec3 v1{1.0f, 0.0f, 0.0f};
   const rt::Vec3 v2{0.0f, 1.0f, 0.0f};

   SECTION("touché") {
      const rt::Ray ray{rt::Vec3{0.25f, 0.25f, 1.0f}, rt::Vec3{0.0f, 0.0f, -1.0f}};
      rt::TriangleHit hit{};
      REQUIRE(rt::intersect_triangle(ray, v0, v1, v2, hit));
      CHECK(hit.t == Catch::Approx(1.0f));
      CHECK(hit.u == Catch::Approx(0.25f));
      CHECK(hit.v == Catch::Approx(0.25f));
   }

   SECTION("raté : point hors du triangle") {
      const rt::Ray ray{rt::Vec3{1.0f, 1.0f, 1.0f}, rt::Vec3{0.0f, 0.0f, -1.0f}};
      rt::TriangleHit hit{};
      REQUIRE_FALSE(rt::intersect_triangle(ray, v0, v1, v2, hit));
   }

   SECTION("raté : triangle derrière le rayon") {
      const rt::Ray ray{rt::Vec3{0.25f, 0.25f, -1.0f}, rt::Vec3{0.0f, 0.0f, -1.0f}};
      rt::TriangleHit hit{};
      REQUIRE_FALSE(rt::intersect_triangle(ray, v0, v1, v2, hit));
   }

   SECTION("raté : rayon parallèle au triangle") {
      const rt::Ray ray{rt::Vec3{0.25f, 0.25f, 1.0f}, rt::Vec3{1.0f, 0.0f, 0.0f}};
      rt::TriangleHit hit{};
      REQUIRE_FALSE(rt::intersect_triangle(ray, v0, v1, v2, hit));
   }

   SECTION("raté : rayon parallèle dans le plan du triangle") {
      const rt::Ray ray{rt::Vec3{-1.0f, 0.25f, 0.0f}, rt::Vec3{1.0f, 0.0f, 0.0f}};
      rt::TriangleHit hit{};
      REQUIRE_FALSE(rt::intersect_triangle(ray, v0, v1, v2, hit));
   }
}

TEST_CASE("intersection d'un triangle à la compilation", "[triangle]") {
   constexpr rt::Ray kRay{rt::Vec3{0.25f, 0.25f, 1.0f}, rt::Vec3{0.0f, 0.0f, -1.0f}};
   constexpr auto kHits = [](rt::Ray ray) {
      rt::TriangleHit hit{};
      return rt::intersect_triangle(ray, {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, hit);
   };
   STATIC_REQUIRE(kHits(kRay));
}
