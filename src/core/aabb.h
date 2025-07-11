#pragma once

#include "core/host_device.h"
#include "core/ray.h"
#include "core/vec3.h"

namespace rt {

// Boîte englobante alignée sur les axes. Vide par défaut : min = +inf, max = -inf
struct Aabb {
   Vec3 min{kInfinity, kInfinity, kInfinity};
   Vec3 max{-kInfinity, -kInfinity, -kInfinity};

   RT_HOST_DEVICE constexpr void grow(Vec3 p) {
      min = rt::min(min, p);
      max = rt::max(max, p);
   }
};

// Méthode des dalles. inv_dir = 1 / direction, composante par composante
RT_HOST_DEVICE constexpr bool intersect_aabb(Ray ray, Vec3 inv_dir, Aabb box, float t_max) {
   const Vec3 t0 = (box.min - ray.origin) * inv_dir;
   const Vec3 t1 = (box.max - ray.origin) * inv_dir;
   const float t_enter = max_component(min(t0, t1));
   const float t_exit = min_component(max(t0, t1));
   return t_enter <= t_exit && t_exit > 0.f && t_enter < t_max;
}

} // namespace rt
