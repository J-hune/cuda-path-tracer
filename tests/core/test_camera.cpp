#include "core/camera.h"
#include "core/ray.h"
#include "core/vec2.h"
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

TEST_CASE("lentille : tous les rayons d'un pixel passent par le même point net", "[camera]") {
   using Catch::Matchers::WithinAbs;
   constexpr rt::CameraSettings kLens{
      .vertical_fov_degrees = 90.f,
      .aperture_radius = 0.3f,
      .focus_distance = 4.f,
   };
   const rt::Camera camera{kLens, kWidth, kHeight};
   const rt::Camera pinhole{kSettings, kWidth, kHeight};
   const rt::Ray center = pinhole.generate_ray(37.f, 81.f);
   // Point net : sur le rayon du sténopé, à 4 le long de l'axe de visée (-z)
   const rt::Vec3 focus = center.at(4.f / -center.direction.z);

   for (const rt::Vec2 lens : {rt::Vec2{0.1f, 0.2f}, rt::Vec2{0.9f, 0.5f}, rt::Vec2{0.4f, 0.95f}}) {
      const rt::Ray ray = camera.generate_ray(37.f, 81.f, lens);
      // Origine sur la lentille : dans le plan de la caméra, à moins du rayon d'ouverture
      CHECK_THAT(ray.origin.z, WithinAbs(0.0, 1e-6));
      CHECK(rt::length(ray.origin) <= 0.3f + 1e-6f);
      const rt::Vec3 hit = ray.at(4.f / -ray.direction.z);
      CHECK_THAT(hit.x, WithinAbs(focus.x, 1e-4));
      CHECK_THAT(hit.y, WithinAbs(focus.y, 1e-4));
   }

   // Centre de la lentille : le rayon du sténopé
   const rt::Ray middle = camera.generate_ray(37.f, 81.f);
   CHECK(rt::length(middle.origin) < 1e-6f);
   CHECK_THAT(rt::dot(middle.direction, center.direction), WithinAbs(1.0, 1e-6));
}
