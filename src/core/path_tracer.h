#pragma once

#include "core/host_device.h"
#include "core/material.h"
#include "core/ray.h"
#include "core/rng.h"
#include "core/sampling.h"
#include "core/scene_view.h"
#include "core/texture.h"
#include "core/vec2.h"
#include "core/vec3.h"

#include <cmath>
#include <cstdint>

namespace rt {

inline constexpr int kRouletteStart = 3;
inline constexpr int kMaxBounces = 64;
inline constexpr float kRayOffset = 1e-3f;

// Normale géométrique du triangle, orientée vers le côté d'où vient le rayon
RT_HOST_DEVICE inline Vec3 facing_normal(SceneView view, std::uint32_t triangle, Vec3 direction) {
   const Vec3 v0 = view.positions[view.indices[(3 * triangle) + 0]];
   const Vec3 v1 = view.positions[view.indices[(3 * triangle) + 1]];
   const Vec3 v2 = view.positions[view.indices[(3 * triangle) + 2]];
   const Vec3 n = normalize(cross(v1 - v0, v2 - v0));
   return dot(n, direction) > 0.0f ? -n : n;
}

// Vérifie si un objet se trouve entre l'origine et la cible
RT_HOST_DEVICE inline bool is_occluded(SceneView view, Vec3 origin, Vec3 target) {
   const Vec3 direction = target - origin;
   const float distance = length(direction);
   const Ray ray{origin, normalize(direction)};
   Hit hit{};

   // La lampe visée ne compte pas comme obstacle
   return intersect_scene(ray, view, hit) && hit.t < distance - kRayOffset;
}

// Échantillonne un triangle émissif au hasard et calcule la contribution de la lumière directe
RT_HOST_DEVICE inline Vec3 direct_light(SceneView view, Vec3 x, Vec3 n, Vec3 albedo, Pcg32& rng) {
   if (view.emissive_triangle_count == 0) {
      return Vec3{0.0f, 0.0f, 0.0f};
   }

   // Choix uniforme d'une lampe, index borné à count - 1
   const auto picked = static_cast<std::uint32_t>(
      rng.next_float() * static_cast<float>(view.emissive_triangle_count)
   );
   const std::uint32_t k = picked < view.emissive_triangle_count ? picked
                                                                 : view.emissive_triangle_count - 1;
   const std::uint32_t tri = view.emissive_triangles[k];
   const Vec3 v0 = view.positions[view.indices[(3 * tri) + 0]];
   const Vec3 v1 = view.positions[view.indices[(3 * tri) + 1]];
   const Vec3 v2 = view.positions[view.indices[(3 * tri) + 2]];
   const Vec3 y = sample_point_on_triangle(v0, v1, v2, rng);

   const Vec3 cross_l = cross(v1 - v0, v2 - v0);
   const float area = 0.5f * length(cross_l);
   const Vec3 n_l = normalize(cross_l);

   const float d2 = length_squared(y - x);
   const Vec3 wi = (y - x) / std::sqrt(d2);

   const float cos_x = dot(n, wi);
   const float cos_y = std::fabs(dot(n_l, wi));

   if (cos_x <= 0.0f || cos_y <= 0.0f || is_occluded(view, x + (n * kRayOffset), y)) {
      return Vec3{0.0f, 0.0f, 0.0f};
   }

   const Material material = view.materials[view.material_indices[tri]];
   // BSDF lambertienne = albedo / π ; densité du point tiré = 1 / (aire × nombre de lampes)
   const float geometry = cos_x * cos_y / d2;
   const float inverse_pdf = area * static_cast<float>(view.emissive_triangle_count);
   return (albedo / kPi) * material.emission * (geometry * inverse_pdf);
}

// Coordonnées de texture au point touché, interpolées entre les trois sommets
RT_HOST_DEVICE inline Vec2 interpolate_uv(SceneView view, const Hit& hit) {
   const Vec2 uv0 = view.uvs[view.indices[(3 * hit.triangle) + 0]];
   const Vec2 uv1 = view.uvs[view.indices[(3 * hit.triangle) + 1]];
   const Vec2 uv2 = view.uvs[view.indices[(3 * hit.triangle) + 2]];
   return ((1.0f - hit.u - hit.v) * uv0) + (hit.u * uv1) + (hit.v * uv2);
}

// Albédo du matériau au point touché : couleur de base, multipliée par la texture s'il y en a une
RT_HOST_DEVICE inline Vec3 surface_albedo(
   SceneView view,
   const Material& material,
   const Hit& hit
) {
   if (material.base_color_texture < 0) {
      return material.albedo;
   }
   const Texture texture = view.textures[material.base_color_texture];
   return material.albedo * sample_texture(texture, view.texels, interpolate_uv(view, hit));
}

// Suit un chemin à partir du rayon caméra et renvoie la lumière qu'il rapporte
RT_HOST_DEVICE inline Vec3 trace_path(Ray ray, SceneView view, Pcg32& rng) {
   Vec3 radiance{0.0f, 0.0f, 0.0f};
   Vec3 throughput{1.0f, 1.0f, 1.0f};

   for (int bounce = 0; bounce < kMaxBounces; ++bounce) {
      Hit hit{};
      if (!intersect_scene(ray, view, hit)) {
         radiance += throughput * view.environment.radiance(ray.direction);
         return radiance;
      }

      const Material material = view.materials[view.material_indices[hit.triangle]];
      if (bounce == 0) {
         radiance += throughput * material.emission;
      }
      const Vec3 point = ray.at(hit.t);
      const Vec3 n = facing_normal(view, hit.triangle, ray.direction);
      const Vec3 albedo = surface_albedo(view, material, hit);
      const Vec3 direct = direct_light(view, point, n, albedo, rng);
      radiance += throughput * direct;
      throughput = throughput * albedo;

      if (bounce >= kRouletteStart) {
         const float p = std::fmin(max_component(throughput), 0.95f);
         if (rng.next_float() >= p) {
            return radiance;
         }
         throughput /= p;
      }

      ray = {point + (n * kRayOffset), sample_cosine_hemisphere(n, rng)};
   }

   return radiance;
}

} // namespace rt
