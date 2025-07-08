#include "core/camera.h"
#include "core/ray.h"
#include "core/vec3.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

constexpr rt::CameraSettings kSettings{.vertical_fov_degrees = 90.f};
constexpr int kWidth = 200;
constexpr int kHeight = 100;

TEST_CASE("centre de l'image : rayon dans l'axe de visée", "[camera]") {
   const rt::Camera camera{kSettings, kWidth, kHeight};
   const rt::Ray ray = camera.generate_ray(kWidth / 2.f, kHeight / 2.f);
   REQUIRE(ray.origin == kSettings.position);
   REQUIRE(ray.direction == rt::Vec3{0.f, 0.f, -1.f});
}

TEST_CASE("coin haut gauche : rayon vers la gauche et vers le haut", "[camera]") {
   const rt::Camera camera{kSettings, kWidth, kHeight};
   const rt::Ray ray = camera.generate_ray(0.f, 0.f);
   REQUIRE(ray.origin == kSettings.position);
   REQUIRE(ray.direction.x < 0.f);
   REQUIRE(ray.direction.y > 0.f);
}
