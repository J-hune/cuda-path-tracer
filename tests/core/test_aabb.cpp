#include "core/aabb.h"
#include "core/vec3.h"

#include <catch2/catch_test_macros.hpp>
#include <limits>

namespace {
constexpr rt::Aabb kUnitBox{{0, 0, 0}, {1, 1, 1}};
constexpr float kNoLimit = std::numeric_limits<float>::infinity();

constexpr rt::Vec3 inverse(rt::Vec3 v) {
   return {1.f / v.x, 1.f / v.y, 1.f / v.z};
}

constexpr bool hits(rt::Vec3 origin, rt::Vec3 direction, float t_max = kNoLimit) {
   return rt::intersect_aabb({origin, direction}, inverse(direction), kUnitBox, t_max);
}
} // namespace

TEST_CASE("grow d'une boîte vide", "[aabb]") {
   rt::Aabb box;
   box.grow({1, 2, 3});
   CHECK(box.min == rt::Vec3{1, 2, 3});
   CHECK(box.max == rt::Vec3{1, 2, 3});

   box.grow({-1, 5, 0});
   CHECK(box.min == rt::Vec3{-1, 2, 0});
   CHECK(box.max == rt::Vec3{1, 5, 3});
}

TEST_CASE("intersection rayon-boîte", "[aabb]") {
   constexpr rt::Vec3 kDown{0.1f, 0.1f, -1};

   STATIC_REQUIRE(hits({0.5f, 0.5f, 2}, kDown));
   STATIC_REQUIRE_FALSE(hits({2, 2, 2}, kDown));
   STATIC_REQUIRE_FALSE(hits({0.5f, 0.5f, -1}, kDown));
   STATIC_REQUIRE(hits({0.5f, 0.5f, 0.5f}, kDown));
   STATIC_REQUIRE(hits({-1, -1, -1}, {1, 1, 1}));
   STATIC_REQUIRE_FALSE(hits({0.5f, 0.5f, 2}, kDown, 0.5f));
}

TEST_CASE("intersection rayon-boîte, direction parallèle à un axe", "[aabb]") {
   // Composantes nulles : inv_dir contient des infinis
   CHECK(hits({0.5f, 0.5f, 2}, {0, 0, -1}));
   CHECK_FALSE(hits({1.5f, 0.5f, 2}, {0, 0, -1}));
}
