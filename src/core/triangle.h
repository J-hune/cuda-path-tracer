#pragma once
#include "core/host_device.h"
#include "core/ray.h"
#include "core/vec3.h"

#include <cmath>

namespace rt {

struct TriangleHit {
   float t;
   float u;
   float v;
};

// Intersection rayon-triangle de Möller-Trumbore
RT_HOST_DEVICE constexpr bool intersect_triangle(
   Ray ray,
   Vec3 v0,
   Vec3 v1,
   Vec3 v2,
   TriangleHit& hit
) {
   constexpr float kEpsilon = 1e-8f;

   const Vec3 edge1 = v1 - v0;
   const Vec3 edge2 = v2 - v0;
   const Vec3 p = cross(ray.direction, edge2);
   const float det = dot(edge1, p);
   if (std::fabs(det) < kEpsilon) {
      return false;
   }

   const float inv = 1.f / det;
   const Vec3 s = ray.origin - v0;
   const float u = dot(s, p) * inv;
   if (u < 0.f || u > 1.f) {
      return false;
   }

   const Vec3 q = cross(s, edge1);
   const float v = dot(ray.direction, q) * inv;
   if (v < 0.f || u + v > 1.f) {
      return false;
   }

   const float t = dot(edge2, q) * inv;
   if (t <= 0.f) {
      return false;
   }

   hit.t = t;
   hit.u = u;
   hit.v = v;
   return true;
}
} // namespace rt
