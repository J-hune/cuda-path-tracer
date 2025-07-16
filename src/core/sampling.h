#pragma once

#include "core/host_device.h"
#include "core/rng.h"
#include "core/vec3.h"

#include <cmath>

namespace rt {

RT_HOST_DEVICE inline Vec3 random_unit_vector(Pcg32& rng) {
   const float u1 = rng.next_float();
   const float u2 = rng.next_float();

   const float z = 1.f - 2.f * u1;
   const float r = std::sqrt(std::fmax(0.f, 1.f - z * z));
   const float phi = 2.f * kPi * u2;
   const float x = r * std::cos(phi);
   const float y = r * std::sin(phi);
   return Vec3{x, y, z};
}

RT_HOST_DEVICE inline Vec3 sample_cosine_hemisphere(Vec3 normal, Pcg32& rng) {
   const Vec3 direction = normal + random_unit_vector(rng);
   // Vecteur aléatoire presque opposé à la normale : somme quasi nulle
   if (length_squared(direction) < 1e-12f) {
      return normal;
   }
   return normalize(direction);
}

RT_HOST_DEVICE inline Vec3 sample_point_on_triangle(
   const Vec3& v0,
   const Vec3& v1,
   const Vec3& v2,
   Pcg32& rng
) {
   float u = rng.next_float();
   float v = rng.next_float();
   if (u + v > 1.f) {
      u = 1.f - u;
      v = 1.f - v;
   }
   return v0 + u * (v1 - v0) + v * (v2 - v0);
}

} // namespace rt
