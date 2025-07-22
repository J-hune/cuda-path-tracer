#pragma once

#include "core/host_device.h"
#include "core/rng.h"
#include "core/vec2.h"
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

// Mapping concentrique de Shirley et Chiu : carré [0, 1]² -> disque unité, en préservant les
// aires (des points uniformes dans le carré restent uniformes dans le disque) et la proximité
RT_HOST_DEVICE inline Vec2 square_to_disk_concentric(Vec2 u) {
   const float ox = (2.0f * u.x) - 1.0f;
   const float oy = (2.0f * u.y) - 1.0f;
   if (ox == 0.0f && oy == 0.0f) {
      return {0.0f, 0.0f};
   }
   float radius = 0.0f;
   float theta = 0.0f;
   if (std::fabs(ox) > std::fabs(oy)) {
      radius = ox;
      theta = (kPi / 4.0f) * (oy / ox);
   } else {
      radius = oy;
      theta = (kPi / 2.0f) - ((kPi / 4.0f) * (ox / oy));
   }
   return {radius * std::cos(theta), radius * std::sin(theta)};
}

// Méthode de Malley : un point uniforme du disque, projeté sur l'hémisphère, suit une densité
// cos θ / π. Direction dans le repère local (normale = +z)
RT_HOST_DEVICE inline Vec3 square_to_cosine_hemisphere(Vec2 u) {
   const Vec2 d = square_to_disk_concentric(u);
   const float z = std::sqrt(std::fmax(0.0f, 1.0f - (d.x * d.x) - (d.y * d.y)));
   return {d.x, d.y, z};
}

RT_HOST_DEVICE constexpr float cosine_hemisphere_pdf(Vec3 local_direction) {
   return local_direction.z > 0.0f ? local_direction.z / kPi : 0.0f;
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
