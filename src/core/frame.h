#pragma once

#include "core/host_device.h"
#include "core/vec3.h"

#include <cmath>

namespace rt {

// Repère orthonormé direct (tangente, bitangente, normale). Les BSDF travaillent dans ce repère
// local, où la normale est l'axe z : cos θ d'une direction est simplement sa composante z.
struct Frame {
   Vec3 tangent;
   Vec3 bitangent;
   Vec3 normal;

   [[nodiscard]] RT_HOST_DEVICE constexpr Vec3 to_local(Vec3 v) const {
      return {dot(v, tangent), dot(v, bitangent), dot(v, normal)};
   }

   [[nodiscard]] RT_HOST_DEVICE constexpr Vec3 to_world(Vec3 v) const {
      return (v.x * tangent) + (v.y * bitangent) + (v.z * normal);
   }
};

// Repère construit autour d'une normale unitaire, sans division instable
// (Duff et al., « Building an Orthonormal Basis, Revisited », JCGT 2017)
RT_HOST_DEVICE inline Frame make_frame(Vec3 normal) {
   const float sign = std::copysign(1.0f, normal.z);
   const float a = -1.0f / (sign + normal.z);
   const float b = normal.x * normal.y * a;
   return {
      .tangent = {1.0f + (sign * normal.x * normal.x * a), sign * b, -sign * normal.x},
      .bitangent = {b, sign + (normal.y * normal.y * a), -normal.y},
      .normal = normal,
   };
}

} // namespace rt
