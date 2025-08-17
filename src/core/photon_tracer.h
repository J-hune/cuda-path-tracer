#pragma once

#include "core/aabb.h"
#include "core/bsdf.h"
#include "core/frame.h"
#include "core/host_device.h"
#include "core/light.h"
#include "core/material.h"
#include "core/path_tracer.h"
#include "core/photon_map.h"
#include "core/ray.h"
#include "core/rng.h"
#include "core/sampling.h"
#include "core/scene_view.h"
#include "core/vec2.h"
#include "core/vec3.h"

#include <cmath>
#include <cstdint>

namespace rt {

struct PhotonRay {
   Ray ray;
   Vec3 power;
};

// Direction uniforme dans le cône d'axe axis et de demi-angle θ, avec one_minus_cos = 1 - cos θ
RT_HOST_DEVICE inline Vec3 sample_cone(Vec3 axis, float one_minus_cos, Vec2 u) {
   const float cos_theta = 1.0f - (u.x * one_minus_cos);
   const float sin_theta = std::sqrt(std::fmax(0.0f, 1.0f - (cos_theta * cos_theta)));
   const float phi = 2.0f * kPi * u.y;
   return make_frame(axis).to_world(
      {std::cos(phi) * sin_theta, std::sin(phi) * sin_theta, cos_theta}
   );
}

// Cône des directions qui partent de origin et touchent la sphère d'une cible
struct TargetCone {
   Vec3 axis;
   float one_minus_cos = 2.0f; // 1 - cos du demi-angle, 2 : toutes les directions
};

RT_HOST_DEVICE inline TargetCone target_cone(const PhotonTarget& target, Vec3 origin) {
   const Vec3 to_center = target.center - origin;
   const float d2 = length_squared(to_center);
   const float r2 = target.radius * target.radius;
   if (d2 <= r2) {
      return {.axis = {0.0f, 0.0f, 1.0f}, .one_minus_cos = 2.0f};
   }
   // 1 - cos θ = sin² θ / (1 + cos θ), sans perte de précision pour les cibles lointaines
   const float sin2 = r2 / d2;
   return {
      .axis = to_center / std::sqrt(d2),
      .one_minus_cos = sin2 / (1.0f + std::sqrt(1.0f - sin2)),
   };
}

RT_HOST_DEVICE inline bool in_cone(const TargetCone& cone, Vec3 direction) {
   return 1.0f - dot(cone.axis, direction) <= cone.one_minus_cos;
}

struct TargetedDirection {
   Vec3 direction;
   float pdf = 0.0f; // par angle solide
};

// Direction vers une cible tirée selon son angle solide. Les cônes peuvent se recouvrir : la
// densité est celle du mélange, qui compte tous les cônes contenant la direction.
RT_HOST_DEVICE inline TargetedDirection sample_targets(SceneView view, Vec3 origin, Pcg32& rng) {
   float total = 0.0f;
   for (std::uint32_t i = 0; i < view.photon_target_count; ++i) {
      total += target_cone(view.photon_targets[i], origin).one_minus_cos;
   }
   float u = rng.next_float() * total;
   TargetCone chosen = target_cone(view.photon_targets[0], origin);
   for (std::uint32_t i = 0; i < view.photon_target_count; ++i) {
      chosen = target_cone(view.photon_targets[i], origin);
      if (u < chosen.one_minus_cos) {
         break;
      }
      u -= chosen.one_minus_cos;
   }
   const Vec3 direction = sample_cone(
      chosen.axis,
      chosen.one_minus_cos,
      {rng.next_float(), rng.next_float()}
   );
   float covering = 0.0f;
   for (std::uint32_t i = 0; i < view.photon_target_count; ++i) {
      covering += in_cone(target_cone(view.photon_targets[i], origin), direction) ? 1.0f : 0.0f;
   }
   // Angle solide d'un cône : 2π (1 - cos θ)
   return {.direction = direction, .pdf = std::fmax(covering, 1.0f) / (2.0f * kPi * total)};
}

// Photon émis par une lumière choisie selon la répartition des lumières. Sa puissance est le flux
// de la lumière divisé par la probabilité de l'avoir tirée, pour un seul photon : diviser ensuite
// par le nombre émis. L'HDRI n'émet pas de photons : ses caustiques restent au path tracer.
RT_HOST_DEVICE inline PhotonRay emit_photon(SceneView view, Pcg32& rng) {
   float others = 1.0f; // probabilité totale des lumières autres que l'HDRI
   if (view.environment_map.present()) {
      const std::uint32_t k_env = environment_light_index(view);
      others = view.light_cdf != nullptr
                  ? view.light_cdf[k_env]
                  : static_cast<float>(k_env) / static_cast<float>(light_count(view));
   }
   if (others <= 0.0f) {
      return {.ray = {{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}, .power = {0.0f, 0.0f, 0.0f}};
   }
   const LightChoice choice = pick_light(view, rng.next_float() * others);
   const std::uint32_t k = choice.index;
   const float light_choice = others / choice.probability;

   if (k < view.emissive_triangle_count) {
      // Triangle émissif, selon une loi en cosinus, des deux côtés ou du côté de sa normale
      const std::uint32_t triangle = view.emissive_triangles[k];
      const Vec3 v0 = view.positions[view.indices[(3 * triangle) + 0]];
      const Vec3 v1 = view.positions[view.indices[(3 * triangle) + 1]];
      const Vec3 v2 = view.positions[view.indices[(3 * triangle) + 2]];
      const Vec2 b = sample_triangle_barycentrics(rng);
      const Vec3 y = v0 + (b.x * (v1 - v0)) + (b.y * (v2 - v0));
      const Vec3 cross_l = cross(v1 - v0, v2 - v0);
      const float area = 0.5f * length(cross_l);
      const Material material = view.materials[view.material_indices[triangle]];
      const Vec3 emission = surface_emission(view, material, triangle_uv(view, triangle, b.x, b.y));
      if (view.photon_target_count > 0) {
         const TargetedDirection target = sample_targets(view, y, rng);
         const Vec3 n_l = normalize(cross_l);
         const float cos_l = dot(n_l, target.direction);
         const float emitted = material.double_sided ? std::fabs(cos_l) : std::fmax(cos_l, 0.0f);
         return {
            .ray = {y + (n_l * (cos_l > 0.0f ? kRayOffset : -kRayOffset)), target.direction},
            .power = emission * (emitted * area * light_choice / target.pdf),
         };
      }
      Vec3 side = normalize(cross_l);
      float sides = 1.0f;
      if (material.double_sided) {
         side = rng.next_float() < 0.5f ? side : -side;
         sides = 2.0f;
      }
      const Vec3 direction = make_frame(side).to_world(
         square_to_cosine_hemisphere({rng.next_float(), rng.next_float()})
      );
      return {
         .ray = {y + (side * kRayOffset), direction},
         .power = emission * (sides * kPi * area * light_choice),
      };
   }

   const PunctualLight& light = view.punctual_lights[k - view.emissive_triangle_count];
   if (view.photon_target_count > 0 && light.type != PunctualLight::Type::Directional) {
      const TargetedDirection target = sample_targets(view, light.position, rng);
      const float attenuation = light.type == PunctualLight::Type::Spot
                                   ? spot_attenuation(light, target.direction)
                                   : 1.0f;
      return {
         .ray = {light.position, target.direction},
         .power = light.intensity * (attenuation * light_choice / target.pdf),
      };
   }
   const Vec2 u{rng.next_float(), rng.next_float()};
   switch (light.type) {
      case PunctualLight::Type::Point:
         return {
            .ray = {light.position, random_unit_vector(rng)},
            .power = light.intensity * (4.0f * kPi * light_choice),
         };
      case PunctualLight::Type::Spot: {
         const Vec3 direction = sample_cone(light.direction, 1.0f - light.cos_outer, u);
         const float solid_angle = 2.0f * kPi * (1.0f - light.cos_outer);
         return {
            .ray = {light.position, direction},
            .power = light.intensity *
                     (spot_attenuation(light, direction) * solid_angle * light_choice),
         };
      }
      case PunctualLight::Type::Directional:
         break;
   }
   // Lumière directionnelle : disque perpendiculaire couvrant la sphère englobant la scène
   const Aabb bounds = view.nodes[0].bounds;
   const Vec3 center = 0.5f * (bounds.min + bounds.max);
   const float radius = 0.5f * length(bounds.max - bounds.min);
   const Vec2 d = square_to_disk_concentric(u) * radius;
   const Vec3 origin = center - (light.direction * (2.0f * radius)) +
                       make_frame(light.direction).to_world({d.x, d.y, 0.0f});
   return {
      .ray = {origin, light.direction},
      .power = light.intensity * (kPi * radius * radius * light_choice),
   };
}

// Suit un photon jusqu'à la première surface diffuse. Il y est stocké s'il a traversé du verre ou
// rebondi sur du métal poli avant : sinon c'est de l'éclairage direct, calculé par le path tracer.
RT_HOST_DEVICE inline bool trace_photon(
   SceneView view,
   PhotonRay emitted,
   Pcg32& rng,
   Photon& stored
) {
   Ray ray = emitted.ray;
   Vec3 throughput{1.0f, 1.0f, 1.0f};
   float wavelength = 0.0f;
   for (int bounce = 0; bounce < kMaxBounces; ++bounce) {
      Hit hit{};
      if (!intersect_scene(ray, view, hit)) {
         return false;
      }
      const Material material = view.materials[view.material_indices[hit.triangle]];
      const Vec2 uv = triangle_uv(view, hit.triangle, hit.u, hit.v);
      const SurfaceParams params = surface_params(view, material, uv);
      const Vec3 albedo = params.base_color;

      if (material.transmission > 0.0f && rng.next_float() < material.transmission) {
         ray = scatter_glass(view, hit, material, albedo, ray, rng, throughput, wavelength);
      } else if (is_glossy_metal(material)) {
         const ShadingPoint shading = make_shading_point(view, hit, material, params, uv, ray);
         const BsdfSample bsdf = principled::sample(
            shading.params,
            shading.wo,
            {rng.next_float(), rng.next_float()}
         );
         const Vec3 wi = shading.frame.to_world(bsdf.wi);
         if (bsdf.pdf <= 0.0f || dot(wi, shading.geometric_normal) <= 0.0f) {
            return false;
         }
         ray = {shading.position + (shading.geometric_normal * kRayOffset), wi};
         throughput = throughput * bsdf.weight;
      } else {
         if (bounce == 0) {
            return false;
         }
         stored = {
            .position = ray.at(hit.t),
            .power = emitted.power * throughput,
            .direction = -ray.direction,
            .normal = facing_normal(view, hit.triangle, ray.direction),
         };
         return true;
      }

      if (bounce >= kRouletteStart) {
         const float p = std::fmin(max_component(throughput), 0.95f);
         if (rng.next_float() >= p) {
            return false;
         }
         throughput /= p;
      }
   }
   return false;
}

} // namespace rt
