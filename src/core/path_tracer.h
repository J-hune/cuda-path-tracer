#pragma once

#include "core/bsdf.h"
#include "core/dielectric.h"
#include "core/frame.h"
#include "core/host_device.h"
#include "core/light.h"
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
// Distance jusqu'à laquelle on cherche un obstacle vers une lumière directionnelle (à l'infini)
inline constexpr float kDirectionalLightDistance = 1e6f;

// Normale géométrique du triangle, dans le sens donné par l'ordre des sommets (vers l'extérieur
// pour un maillage fermé orienté comme le veut glTF)
RT_HOST_DEVICE inline Vec3 geometric_normal(SceneView view, std::uint32_t triangle) {
   const Vec3 v0 = view.positions[view.indices[(3 * triangle) + 0]];
   const Vec3 v1 = view.positions[view.indices[(3 * triangle) + 1]];
   const Vec3 v2 = view.positions[view.indices[(3 * triangle) + 2]];
   return normalize(cross(v1 - v0, v2 - v0));
}

// Normale géométrique du triangle, orientée vers le côté d'où vient le rayon
RT_HOST_DEVICE inline Vec3 facing_normal(SceneView view, std::uint32_t triangle, Vec3 direction) {
   const Vec3 n = geometric_normal(view, triangle);
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
// Heuristique de puissance de Veach (β = 2) : poids MIS de la stratégie de densité a face à b
RT_HOST_DEVICE inline float power_heuristic(float a, float b) {
   const float a2 = a * a;
   const float b2 = b * b;
   return a2 + b2 > 0.0f ? a2 / (a2 + b2) : 0.0f;
}

// Nombre de lumières parmi lesquelles l'éclairage direct choisit : triangles émissifs et
// ponctuelles
RT_HOST_DEVICE constexpr std::uint32_t light_count(const SceneView& view) {
   return view.emissive_triangle_count + view.punctual_light_count;
}

// Densité, par unité d'angle solide vue depuis le point éclairé, de tirer ce point d'une lampe
// avec direct_light : 1 / (aire × nombre de lumières), convertie par distance² / cos θ_lampe
RT_HOST_DEVICE inline float light_pdf(
   SceneView view,
   std::uint32_t triangle,
   Vec3 direction,
   float distance
) {
   const Vec3 v0 = view.positions[view.indices[(3 * triangle) + 0]];
   const Vec3 v1 = view.positions[view.indices[(3 * triangle) + 1]];
   const Vec3 v2 = view.positions[view.indices[(3 * triangle) + 2]];
   const Vec3 cross_l = cross(v1 - v0, v2 - v0);
   const float area = 0.5f * length(cross_l);
   const float cos_light = std::fabs(dot(normalize(cross_l), direction));
   if (cos_light <= 0.0f || area <= 0.0f) {
      return 0.0f;
   }
   return distance * distance / (cos_light * area * static_cast<float>(light_count(view)));
}

// Surface au point x : repère local (normale orientée vers l'observateur), direction d'observation
// wo dans ce repère, et paramètres de la BSDF
struct ShadingPoint {
   Vec3 position;
   Frame frame;
   Vec3 wo;
   SurfaceParams params;
};

// Éclairage direct par un triangle émissif, pondéré par MIS face au tirage de la BSDF
RT_HOST_DEVICE inline Vec3 direct_from_triangle(
   SceneView view,
   const ShadingPoint& shading,
   std::uint32_t triangle,
   Pcg32& rng,
   std::uint32_t& ray_count
) {
   const Vec3 v0 = view.positions[view.indices[(3 * triangle) + 0]];
   const Vec3 v1 = view.positions[view.indices[(3 * triangle) + 1]];
   const Vec3 v2 = view.positions[view.indices[(3 * triangle) + 2]];
   const Vec3 y = sample_point_on_triangle(v0, v1, v2, rng);

   const Vec3 cross_l = cross(v1 - v0, v2 - v0);
   const float area = 0.5f * length(cross_l);
   const Vec3 n_l = normalize(cross_l);

   const Vec3 x = shading.position;
   const Vec3 n = shading.frame.normal;
   const float d2 = length_squared(y - x);
   const Vec3 wi = (y - x) / std::sqrt(d2);

   const float cos_x = dot(n, wi);
   const float cos_y = std::fabs(dot(n_l, wi));
   if (cos_x <= 0.0f || cos_y <= 0.0f) {
      return Vec3{0.0f, 0.0f, 0.0f};
   }

   const Vec3 f = principled::eval(shading.params, shading.wo, shading.frame.to_local(wi));
   if (max_component(f) <= 0.0f) {
      return Vec3{0.0f, 0.0f, 0.0f};
   }
   ++ray_count;
   if (is_occluded(view, x + (n * kRayOffset), y)) {
      return Vec3{0.0f, 0.0f, 0.0f};
   }

   const Material material = view.materials[view.material_indices[triangle]];
   // Densité du point tiré = 1 / (aire × nombre de lumières), ramenée en angle solide par d² /
   // cos_y
   const float pdf_light = d2 / (cos_y * area * static_cast<float>(light_count(view)));
   const float bsdf_pdf = principled::pdf(shading.params, shading.wo, shading.frame.to_local(wi));
   const float weight = view.use_mis ? power_heuristic(pdf_light, bsdf_pdf) : 1.0f;
   return f * material.emission * (cos_x * weight / pdf_light);
}

// Éclairage direct par une lumière ponctuelle. Sans surface, elle ne peut pas être touchée par un
// rebond : pas de MIS, toute la contribution vient d'ici
RT_HOST_DEVICE inline Vec3 direct_from_punctual(
   SceneView view,
   const ShadingPoint& shading,
   const PunctualLight& light,
   std::uint32_t& ray_count
) {
   const Vec3 x = shading.position;
   const Vec3 n = shading.frame.normal;

   Vec3 wi{0.0f, 0.0f, 0.0f};
   Vec3 irradiance{0.0f, 0.0f, 0.0f}; // éclairement reçu perpendiculairement à wi
   Vec3 target{0.0f, 0.0f, 0.0f};
   if (light.type == PunctualLight::Type::Directional) {
      wi = -light.direction;
      irradiance = light.intensity;
      target = x + (wi * kDirectionalLightDistance);
   } else {
      const Vec3 to_light = light.position - x;
      const float d2 = length_squared(to_light);
      wi = to_light / std::sqrt(d2);
      irradiance = light.intensity / d2;
      if (light.type == PunctualLight::Type::Spot) {
         irradiance *= spot_attenuation(light, -wi);
      }
      target = light.position;
   }

   const float cos_x = dot(n, wi);
   if (cos_x <= 0.0f || max_component(irradiance) <= 0.0f) {
      return Vec3{0.0f, 0.0f, 0.0f};
   }
   const Vec3 f = principled::eval(shading.params, shading.wo, shading.frame.to_local(wi));
   if (max_component(f) <= 0.0f) {
      return Vec3{0.0f, 0.0f, 0.0f};
   }
   ++ray_count;
   if (is_occluded(view, x + (n * kRayOffset), target)) {
      return Vec3{0.0f, 0.0f, 0.0f};
   }
   // Lumière choisie avec la probabilité 1 / nombre de lumières
   return f * irradiance * (cos_x * static_cast<float>(light_count(view)));
}

// Éclairage direct : choisit uniformément une lumière (triangle émissif ou ponctuelle).
// ray_count est incrémenté du nombre de rayons lancés (le rayon d'ombre)
RT_HOST_DEVICE inline Vec3 direct_light(
   SceneView view,
   const ShadingPoint& shading,
   Pcg32& rng,
   std::uint32_t& ray_count
) {
   const std::uint32_t count = light_count(view);
   if (count == 0) {
      return Vec3{0.0f, 0.0f, 0.0f};
   }
   // Choix uniforme, index borné à count - 1 (l'arrondi en float peut donner count)
   const auto picked = static_cast<std::uint32_t>(rng.next_float() * static_cast<float>(count));
   const std::uint32_t k = picked < count ? picked : count - 1;
   if (k < view.emissive_triangle_count) {
      return direct_from_triangle(view, shading, view.emissive_triangles[k], rng, ray_count);
   }
   const PunctualLight& light = view.punctual_lights[k - view.emissive_triangle_count];
   return direct_from_punctual(view, shading, light, ray_count);
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

// Suit un chemin à partir du rayon caméra et renvoie la lumière qu'il rapporte.
// ray_count est incrémenté du nombre de rayons lancés (rebonds et rayons d'ombre)
RT_HOST_DEVICE inline Vec3 trace_path(
   Ray ray,
   SceneView view,
   Pcg32& rng,
   std::uint32_t& ray_count
) {
   Vec3 radiance{0.0f, 0.0f, 0.0f};
   Vec3 throughput{1.0f, 1.0f, 1.0f};
   // Après un rebond spéculaire, l'éclairage direct n'a pas été calculé : l'émission touchée compte
   // entièrement. Après un rebond de la BSDF, elle est pondérée par MIS face à l'éclairage direct.
   bool specular_bounce = false;
   float bsdf_pdf = 0.0f;

   for (int bounce = 0; bounce < kMaxBounces; ++bounce) {
      Hit hit{};
      ++ray_count;
      if (!intersect_scene(ray, view, hit)) {
         radiance += throughput * view.environment.radiance(ray.direction);
         return radiance;
      }

      const Material material = view.materials[view.material_indices[hit.triangle]];
      if (max_component(material.emission) > 0.0f) {
         if (bounce == 0 || specular_bounce) {
            radiance += throughput * material.emission;
         } else if (view.use_mis) {
            const float pdf_light = light_pdf(view, hit.triangle, ray.direction, hit.t);
            radiance += throughput * material.emission * power_heuristic(bsdf_pdf, pdf_light);
         }
      }
      const Vec3 point = ray.at(hit.t);
      const Vec3 albedo = surface_albedo(view, material, hit);

      if (material.transmission > 0.0f && rng.next_float() < material.transmission) {
         // Surface diélectrique : réflexion ou réfraction, sans éclairage direct
         const Vec3 outward = geometric_normal(view, hit.triangle);
         if (dot(ray.direction, outward) > 0.0f) {
            // Le rayon sort de l'objet : il a traversé le volume sur une distance hit.t
            throughput = throughput * beer_lambert(material.absorption, hit.t);
         }
         const DielectricSample
            sample = sample_dielectric(ray.direction, outward, material.ior, rng);
         if (sample.transmitted) {
            throughput = throughput * albedo;
         }
         const Vec3 side = dot(sample.direction, outward) > 0.0f ? outward : -outward;
         ray = {point + (side * kRayOffset), sample.direction};
         specular_bounce = true;
      } else {
         // Surface opaque (glTF metallic-roughness) : éclairage direct, puis rebond selon la BSDF
         const Vec3 n = facing_normal(view, hit.triangle, ray.direction);
         const Frame frame = make_frame(n);
         const ShadingPoint shading{
            .position = point,
            .frame = frame,
            .wo = frame.to_local(-ray.direction),
            .params = {
               .base_color = albedo,
               .metallic = material.metallic,
               .roughness = material.roughness
            },
         };
         radiance += throughput * direct_light(view, shading, rng, ray_count);

         const BsdfSample bsdf = principled::sample(
            shading.params,
            shading.wo,
            {rng.next_float(), rng.next_float()}
         );
         if (bsdf.pdf <= 0.0f) {
            return radiance;
         }
         ray = {point + (n * kRayOffset), frame.to_world(bsdf.wi)};
         throughput = throughput * bsdf.weight;
         bsdf_pdf = bsdf.pdf;
         specular_bounce = false;
      }

      if (bounce >= kRouletteStart) {
         const float p = std::fmin(max_component(throughput), 0.95f);
         if (rng.next_float() >= p) {
            return radiance;
         }
         throughput /= p;
      }
   }

   return radiance;
}

} // namespace rt
