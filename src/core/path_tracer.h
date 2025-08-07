#pragma once

#include "core/bsdf.h"
#include "core/dielectric.h"
#include "core/frame.h"
#include "core/host_device.h"
#include "core/light.h"
#include "core/material.h"
#include "core/photon_map.h"
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

// Normale d'ombrage interpolée entre les sommets, du même côté que la normale géométrique donnée.
// Normale géométrique si la scène n'a pas de normales ou si le triangle est en ombrage plat.
RT_HOST_DEVICE inline Vec3 shading_normal(SceneView view, const Hit& hit, Vec3 geometric) {
   if (view.normals == nullptr) {
      return geometric;
   }
   const Vec3 n0 = view.normals[view.indices[(3 * hit.triangle) + 0]];
   const Vec3 n1 = view.normals[view.indices[(3 * hit.triangle) + 1]];
   const Vec3 n2 = view.normals[view.indices[(3 * hit.triangle) + 2]];
   const Vec3 n = ((1.0f - hit.u - hit.v) * n0) + (hit.u * n1) + (hit.v * n2);
   if (length_squared(n) < 1e-12f) {
      return geometric;
   }
   const Vec3 shading = normalize(n);
   return dot(shading, geometric) < 0.0f ? -shading : shading;
}

// Terminateur d'ombre (Chiang et al., « Taming the Shadow Terminator », 2019) : atténue la lumière
// que la normale d'ombrage promet là où la géométrie réelle la reçoit en rasant
RT_HOST_DEVICE inline float shadow_terminator(Vec3 geometric, Vec3 shading, Vec3 wi) {
   const float cos_shading = dot(shading, wi);
   const float cos_between = dot(geometric, shading);
   if (cos_shading <= 0.0f || cos_between <= 0.0f) {
      return 0.0f;
   }
   const float g = std::fmin(1.0f, dot(geometric, wi) / (cos_shading * cos_between));
   if (g <= 0.0f) {
      return 0.0f;
   }
   return (-g * g * g) + (g * g) + g;
}

// Surface au point x : repère local (normale orientée vers l'observateur), direction d'observation
// wo dans ce repère, et paramètres de la BSDF
struct ShadingPoint {
   Vec3 position;
   Vec3 geometric_normal; // orientée vers l'observateur : visibilité et décalage des rayons
   Frame frame;           // autour de la normale d'ombrage : BSDF
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
   const Vec3 n_g = shading.geometric_normal;
   const float d2 = length_squared(y - x);
   const Vec3 wi = (y - x) / std::sqrt(d2);

   const float cos_x = dot(n, wi);
   const float cos_y = std::fabs(dot(n_l, wi));
   if (cos_x <= 0.0f || cos_y <= 0.0f || dot(n_g, wi) <= 0.0f) {
      return Vec3{0.0f, 0.0f, 0.0f};
   }

   const Vec3 f = principled::eval(shading.params, shading.wo, shading.frame.to_local(wi));
   if (max_component(f) <= 0.0f) {
      return Vec3{0.0f, 0.0f, 0.0f};
   }
   ++ray_count;
   if (is_occluded(view, x + (n_g * kRayOffset), y)) {
      return Vec3{0.0f, 0.0f, 0.0f};
   }

   const Material material = view.materials[view.material_indices[triangle]];
   // Densité du point tiré = 1 / (aire × nombre de lumières), ramenée en angle solide par d² /
   // cos_y
   const float pdf_light = d2 / (cos_y * area * static_cast<float>(light_count(view)));
   const float bsdf_pdf = principled::pdf(shading.params, shading.wo, shading.frame.to_local(wi));
   const float weight = view.use_mis ? power_heuristic(pdf_light, bsdf_pdf) : 1.0f;
   const float terminator = shadow_terminator(n_g, n, wi);
   return f * material.emission * (cos_x * weight * terminator / pdf_light);
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
   const Vec3 n_g = shading.geometric_normal;

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
   if (cos_x <= 0.0f || dot(n_g, wi) <= 0.0f || max_component(irradiance) <= 0.0f) {
      return Vec3{0.0f, 0.0f, 0.0f};
   }
   const Vec3 f = principled::eval(shading.params, shading.wo, shading.frame.to_local(wi));
   if (max_component(f) <= 0.0f) {
      return Vec3{0.0f, 0.0f, 0.0f};
   }
   ++ray_count;
   if (is_occluded(view, x + (n_g * kRayOffset), target)) {
      return Vec3{0.0f, 0.0f, 0.0f};
   }
   // Lumière choisie avec la probabilité 1 / nombre de lumières
   const float terminator = shadow_terminator(n_g, n, wi);
   return f * irradiance * (cos_x * terminator * static_cast<float>(light_count(view)));
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

// Réflexion ou réfraction sur une surface diélectrique lisse. Met à jour throughput (absorption
// du volume traversé, couleur transmise) et renvoie le rayon suivant.
RT_HOST_DEVICE inline Ray scatter_glass(
   SceneView view,
   const Hit& hit,
   const Material& material,
   Vec3 albedo,
   const Ray& ray,
   Pcg32& rng,
   Vec3& throughput
) {
   const Vec3 outward = geometric_normal(view, hit.triangle);
   if (dot(ray.direction, outward) > 0.0f) {
      // Le rayon sort de l'objet : il a traversé le volume sur une distance hit.t
      throughput = throughput * beer_lambert(material.absorption, hit.t);
   }
   const Vec3 outward_shading = shading_normal(view, hit, outward);
   const DielectricSample
      sample = sample_dielectric(ray.direction, outward_shading, material.ior, rng);
   if (sample.transmitted) {
      throughput = throughput * albedo;
   }
   const Vec3 side = dot(sample.direction, outward) > 0.0f ? outward : -outward;
   return {ray.at(hit.t) + (side * kRayOffset), sample.direction};
}

// Point d'une surface opaque touché par le rayon, avec ses normales et ses paramètres de BSDF
RT_HOST_DEVICE inline ShadingPoint make_shading_point(
   SceneView view,
   const Hit& hit,
   const Material& material,
   Vec3 albedo,
   const Ray& ray
) {
   const Vec3 n_g = facing_normal(view, hit.triangle, ray.direction);
   // Normale d'ombrage, sauf si l'observateur se retrouve derrière elle
   const Vec3 n_s = shading_normal(view, hit, n_g);
   const Vec3 n = dot(n_s, -ray.direction) > 0.0f ? n_s : n_g;
   const Frame frame = make_frame(n);
   return {
      .position = ray.at(hit.t),
      .geometric_normal = n_g,
      .frame = frame,
      .wo = frame.to_local(-ray.direction),
      .params =
         {.base_color = albedo, .metallic = material.metallic, .roughness = material.roughness},
   };
}

// Rugosité en dessous de laquelle un métal est traité comme un miroir par les photons
inline constexpr float kGlossyRoughness = 0.25f;

// Métal poli : les photons y rebondissent sans y être stockés
RT_HOST_DEVICE inline bool is_glossy_metal(const Material& material) {
   return material.metallic >= 0.99f && material.roughness < kGlossyRoughness;
}

// Éclairement des caustiques au point, estimé par la densité des photons à moins de radius
RT_HOST_DEVICE inline Vec3 gather_caustics(
   const PhotonMapView& map,
   const ShadingPoint& shading,
   float radius
) {
   Vec3 sum{0.0f, 0.0f, 0.0f};
   for_each_photon_near(map, shading.position, radius, [&](const Photon& photon) {
      // Photons d'une autre surface (de l'autre côté d'une paroi fine, d'un coin)
      if (dot(photon.normal, shading.geometric_normal) < 0.9f) {
         return;
      }
      const Vec3 wi = shading.frame.to_local(photon.direction);
      sum += principled::eval(shading.params, shading.wo, wi) * photon.power;
   });
   return sum / (kPi * radius * radius);
}

inline constexpr float kDefaultFireflyClamp = 10.0f;

// Écrêtage des fireflies (biaisé), 0 : désactivé
RT_HOST_DEVICE inline Vec3 clamp_contribution(Vec3 contribution, float max_contribution) {
   const float largest = max_component(contribution);
   if (max_contribution <= 0.0f || largest <= max_contribution) {
      return contribution;
   }
   return contribution * (max_contribution / largest);
}

// Couches auxiliaires du débruitage, relevées au premier point touché par le rayon caméra
struct FirstHit {
   Vec3 albedo{0.0f, 0.0f, 0.0f};
   Vec3 normal{0.0f, 0.0f, 0.0f}; // normale d'ombrage, nulle si le rayon ne touche rien
};

// Suit un chemin à partir du rayon caméra et renvoie la lumière qu'il rapporte.
// ray_count est incrémenté du nombre de rayons lancés (rebonds et rayons d'ombre)
RT_HOST_DEVICE inline Vec3 trace_path(
   Ray ray,
   SceneView view,
   Pcg32& rng,
   std::uint32_t& ray_count,
   FirstHit* first_hit = nullptr
) {
   Vec3 radiance{0.0f, 0.0f, 0.0f};
   Vec3 throughput{1.0f, 1.0f, 1.0f};
   // Après un rebond spéculaire, l'éclairage direct n'a pas été calculé : l'émission touchée compte
   // entièrement. Après un rebond de la BSDF, elle est pondérée par MIS face à l'éclairage direct.
   bool specular_bounce = false;
   float bsdf_pdf = 0.0f;
   // Avec les photons : le chemin est passé par une surface diffuse, puis seulement par du verre ou
   // du métal poli. La lumière qu'il trouve alors est une caustique, déjà apportée à cette surface
   // par les photons.
   const bool photons = view.caustics.bucket_count > 0;
   bool after_diffuse = false;
   bool caustic_path = false;
   float distance = 0.0f; // parcourue depuis la caméra

   for (int bounce = 0; bounce < kMaxBounces; ++bounce) {
      // Seul l'éclairage indirect est écrêté
      const float max_contribution = bounce > 0 ? view.firefly_clamp : 0.0f;
      Hit hit{};
      ++ray_count;
      if (!intersect_scene(ray, view, hit)) {
         if (bounce == 0 && first_hit != nullptr) {
            *first_hit = {.albedo = view.environment.radiance(ray.direction)};
         }
         const Vec3 sky = throughput * view.environment.radiance(ray.direction);
         radiance += clamp_contribution(sky, max_contribution);
         return radiance;
      }

      distance += hit.t;
      const Material material = view.materials[view.material_indices[hit.triangle]];
      if (max_component(material.emission) > 0.0f && !caustic_path) {
         if (bounce == 0 || specular_bounce) {
            radiance += clamp_contribution(throughput * material.emission, max_contribution);
         } else if (view.use_mis) {
            const float pdf_light = light_pdf(view, hit.triangle, ray.direction, hit.t);
            const Vec3 emitted = throughput * material.emission *
                                 power_heuristic(bsdf_pdf, pdf_light);
            radiance += clamp_contribution(emitted, max_contribution);
         }
      }
      const Vec3 albedo = surface_albedo(view, material, hit);
      if (bounce == 0 && first_hit != nullptr) {
         const Vec3 facing = facing_normal(view, hit.triangle, ray.direction);
         *first_hit = {.albedo = albedo, .normal = shading_normal(view, hit, facing)};
      }

      if (material.transmission > 0.0f && rng.next_float() < material.transmission) {
         // Surface diélectrique : réflexion ou réfraction, sans éclairage direct
         ray = scatter_glass(view, hit, material, albedo, ray, rng, throughput);
         specular_bounce = true;
         caustic_path = after_diffuse;
      } else {
         // Surface opaque (glTF metallic-roughness) : éclairage direct, puis rebond selon la BSDF
         const ShadingPoint shading = make_shading_point(view, hit, material, albedo, ray);
         const Vec3 n_g = shading.geometric_normal;
         const Vec3 n = shading.frame.normal;
         const bool glossy = photons && is_glossy_metal(material);
         if (!(glossy && after_diffuse)) {
            const Vec3 direct = throughput * direct_light(view, shading, rng, ray_count);
            radiance += clamp_contribution(direct, max_contribution);
         }
         if (photons && !glossy) {
            // Vu depuis la caméra (à travers du verre ou un miroir au plus), le rayon suit la
            // taille d'un pixel ; après un rebond diffus, le flou ne se voit plus : rayon maximal
            const PhotonMapView& map = view.caustics;
            const float radius = after_diffuse || map.radius_per_distance <= 0.0f
                                    ? map.radius
                                    : std::fmin(map.radius, distance * map.radius_per_distance);
            radiance += throughput * gather_caustics(map, shading, radius);
            after_diffuse = true;
         }
         caustic_path = glossy && after_diffuse;

         const BsdfSample bsdf = principled::sample(
            shading.params,
            shading.wo,
            {rng.next_float(), rng.next_float()}
         );
         const Vec3 wi = shading.frame.to_world(bsdf.wi);
         // Une direction sous la surface réelle ne peut pas repartir de ce côté
         if (bsdf.pdf <= 0.0f || dot(wi, n_g) <= 0.0f) {
            return radiance;
         }
         ray = {shading.position + (n_g * kRayOffset), wi};
         throughput = throughput * bsdf.weight * shadow_terminator(n_g, n, wi);
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
