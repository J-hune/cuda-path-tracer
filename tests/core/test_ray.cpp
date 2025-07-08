#include "core/ray.h"
#include "core/vec3.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("Ray::at() renvoie le point O + t·D", "[ray]") {
   constexpr rt::Ray kRay{rt::Vec3{1, 2, 3}, rt::Vec3{0, 0, 1}};
   STATIC_REQUIRE(kRay.at(0.f) == rt::Vec3{1, 2, 3});
   STATIC_REQUIRE(kRay.at(2.f) == rt::Vec3{1, 2, 5});
   STATIC_REQUIRE(kRay.at(-1.f) == rt::Vec3{1, 2, 2});
}

TEST_CASE("Ray::at() : t est la distance parcourue si la direction est normalisée", "[ray]") {
   constexpr rt::Vec3 kDirection{1, 1, 0};
   const rt::Vec3 normalized_direction = rt::normalize(kDirection);
   const rt::Ray ray{rt::Vec3{0, 0, 0}, normalized_direction};

   const float t = 3.f;
   REQUIRE_THAT(rt::length(ray.at(t) - ray.origin), Catch::Matchers::WithinAbs(t, 1e-5f));
}
