#include "core/frame.h"
#include "core/rng.h"
#include "core/sampling.h"
#include "core/vec3.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>

using Catch::Matchers::WithinAbs;

TEST_CASE("make_frame : repère orthonormé direct autour de la normale", "[frame]") {
   rt::Pcg32 rng(11, 0);
   // Directions aléatoires, plus les cas délicats : normale vers -z (singularité évitée), axes
   std::array<rt::Vec3, 4> special{{{0, 0, -1}, {0, 0, 1}, {1, 0, 0}, {0, -1, 0}}};
   for (int i = 0; i < 1000; ++i) {
      const rt::Vec3 n = i < 4 ? special[static_cast<std::size_t>(i)] : rt::random_unit_vector(rng);
      const rt::Frame frame = rt::make_frame(n);

      REQUIRE_THAT(rt::length(frame.tangent), WithinAbs(1.0, 1e-5));
      REQUIRE_THAT(rt::length(frame.bitangent), WithinAbs(1.0, 1e-5));
      REQUIRE_THAT(rt::dot(frame.tangent, frame.bitangent), WithinAbs(0.0, 1e-5));
      REQUIRE_THAT(rt::dot(frame.tangent, n), WithinAbs(0.0, 1e-5));
      REQUIRE_THAT(rt::dot(frame.bitangent, n), WithinAbs(0.0, 1e-5));
      // Direct : tangente × bitangente = normale
      const rt::Vec3 c = rt::cross(frame.tangent, frame.bitangent);
      REQUIRE_THAT(rt::length(c - n), WithinAbs(0.0, 1e-5));
   }
}

TEST_CASE("Frame : to_world est l'inverse de to_local", "[frame]") {
   const rt::Frame frame = rt::make_frame(rt::normalize({1, 2, 3}));
   const rt::Vec3 v{0.3f, -0.7f, 0.5f};
   const rt::Vec3 back = frame.to_world(frame.to_local(v));
   CHECK_THAT(back.x, WithinAbs(v.x, 1e-5));
   CHECK_THAT(back.y, WithinAbs(v.y, 1e-5));
   CHECK_THAT(back.z, WithinAbs(v.z, 1e-5));
   // La normale devient l'axe z du repère local
   CHECK_THAT(frame.to_local(frame.normal).z, WithinAbs(1.0, 1e-6));
}
