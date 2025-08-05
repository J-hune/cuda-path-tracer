#pragma once

#include "core/aabb.h"
#include "core/bsdf.h"
#include "core/frame.h"
#include "core/host_device.h"
#include "core/light.h"
#include "core/material.h"
#include "core/path_tracer.h"
#include "core/ray.h"
#include "core/rng.h"
#include "core/sampling.h"
#include "core/scene_view.h"
#include "core/vec2.h"
#include "core/vec3.h"

#include <cmath>
#include <cstdint>

namespace rt {

// Photon de caustique : lumière arrivée sur une surface diffuse après au moins un rebond sur du
// verre ou du métal poli
struct Photon {
   Vec3 position;
   Vec3 power;     // flux transporté, en W
   Vec3 direction; // direction d'où vient la lumière
   Vec3 normal;    // normale géométrique, du côté d'arrivée
};

// Rugosité en dessous de laquelle un métal est traité comme un miroir par les photons
inline constexpr float kGlossyRoughness = 0.25f;

// Métal poli : les photons y rebondissent sans y être stockés
RT_HOST_DEVICE inline bool is_glossy_metal(const Material& material) {
   return material.metallic >= 0.99f && material.roughness < kGlossyRoughness;
}

struct PhotonRay {
   Ray ray;
   Vec3 power;
};

// Direction uniforme dans le cône d'axe axis et de demi-angle acos(cos_max)
RT_HOST_DEVICE inline Vec3 sample_cone(Vec3 axis, float cos_max, Vec2 u) {
   const float cos_theta = 1.0f - (u.x * (1.0f - cos_max));
   const float sin_theta = std::sqrt(std::fmax(0.0f, 1.0f - (cos_theta * cos_theta)));
   const float phi = 2.0f * kPi * u.y;
   return make_frame(axis).to_world(
      {std::cos(phi) * sin_theta, std::sin(phi) * sin_theta, cos_theta}
   );
}

// Photon émis par une lumière choisie uniformément. Sa puissance est le flux de la lumière divisé
// par la probabilité de l'avoir tiré, pour un seul photon : diviser ensuite par le nombre émis.
RT_HOST_DEVICE inline PhotonRay emit_photon(SceneView view, Pcg32& rng) {
   const std::uint32_t count = light_count(view);
   const auto picked = static_cast<std::uint32_t>(rng.next_float() * static_cast<float>(count));
   const std::uint32_t k = picked < count ? picked : count - 1;
   const auto light_choice = static_cast<float>(count);

   if (k < view.emissive_triangle_count) {
      // Triangle émissif, émis des deux côtés selon une loi en cosinus
      const std::uint32_t triangle = view.emissive_triangles[k];
      const Vec3 v0 = view.positions[view.indices[(3 * triangle) + 0]];
      const Vec3 v1 = view.positions[view.indices[(3 * triangle) + 1]];
      const Vec3 v2 = view.positions[view.indices[(3 * triangle) + 2]];
      const Vec3 y = sample_point_on_triangle(v0, v1, v2, rng);
      const Vec3 cross_l = cross(v1 - v0, v2 - v0);
      const float area = 0.5f * length(cross_l);
      const Vec3 side = rng.next_float() < 0.5f ? normalize(cross_l) : -normalize(cross_l);
      const Vec3 direction = make_frame(side).to_world(
         square_to_cosine_hemisphere({rng.next_float(), rng.next_float()})
      );
      const Material material = view.materials[view.material_indices[triangle]];
      return {
         .ray = {y + (side * kRayOffset), direction},
         .power = material.emission * (2.0f * kPi * area * light_choice),
      };
   }

   const PunctualLight& light = view.punctual_lights[k - view.emissive_triangle_count];
   const Vec2 u{rng.next_float(), rng.next_float()};
   switch (light.type) {
      case PunctualLight::Type::Point:
         return {
            .ray = {light.position, random_unit_vector(rng)},
            .power = light.intensity * (4.0f * kPi * light_choice),
         };
      case PunctualLight::Type::Spot: {
         const Vec3 direction = sample_cone(light.direction, light.cos_outer, u);
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
   for (int bounce = 0; bounce < kMaxBounces; ++bounce) {
      Hit hit{};
      if (!intersect_scene(ray, view, hit)) {
         return false;
      }
      const Material material = view.materials[view.material_indices[hit.triangle]];
      const Vec3 albedo = surface_albedo(view, material, hit);

      if (material.transmission > 0.0f && rng.next_float() < material.transmission) {
         ray = scatter_glass(view, hit, material, albedo, ray, rng, throughput);
      } else if (is_glossy_metal(material)) {
         const ShadingPoint shading = make_shading_point(view, hit, material, albedo, ray);
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
