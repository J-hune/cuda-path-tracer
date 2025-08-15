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
#include "core/spectrum.h"
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
   // La lampe visée ne compte pas comme obstacle
   return intersects_before(ray, view, distance - kRayOffset);
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

struct LightChoice {
   std::uint32_t index; // triangles émissifs d'abord, puis lumières ponctuelles
   float probability;
};

// Tire une lumière avec u dans [0, 1)
RT_HOST_DEVICE inline LightChoice pick_light(const SceneView& view, float u) {
   const std::uint32_t count = light_count(view);
   if (view.light_cdf == nullptr) {
      const auto picked = static_cast<std::uint32_t>(u * static_cast<float>(count));
      return {
         .index = picked < count ? picked : count - 1,
         .probability = 1.0f / static_cast<float>(count)
      };
   }
   // Dichotomie : light_cdf[low] <= u < light_cdf[high]. Une lumière de probabilité nulle
   // (intervalle vide) n'est jamais tirée.
   std::uint32_t low = 0;
   std::uint32_t high = count;
   while (high - low > 1) {
      const std::uint32_t middle = (low + high) / 2;
      if (u < view.light_cdf[middle]) {
         high = middle;
      } else {
         low = middle;
      }
   }
   return {.index = low, .probability = view.light_cdf[low + 1] - view.light_cdf[low]};
}

// Probabilité que pick_light tire le triangle émissif triangle
RT_HOST_DEVICE inline float triangle_light_probability(
   const SceneView& view,
   std::uint32_t triangle
) {
   if (view.triangle_light == nullptr) {
      return 1.0f / static_cast<float>(light_count(view));
   }
   const std::int32_t k = view.triangle_light[triangle];
   return k < 0 ? 0.0f : view.light_cdf[k + 1] - view.light_cdf[k];
}

// Densité, par unité d'angle solide vue depuis le point éclairé, de tirer ce point d'une lampe
// avec direct_light : probabilité du triangle / aire, convertie par distance² / cos θ_lampe
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
   return triangle_light_probability(view, triangle) * distance * distance / (cos_light * area);
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

// Normale d'ombrage perturbée par la texture de normales du matériau, au point de coordonnées de
// texture uv, du même côté que la normale géométrique donnée. Sans texture ou sans tangentes : la
// normale interpolée.
RT_HOST_DEVICE inline Vec3 mapped_normal(
   SceneView view,
   const Hit& hit,
   const Material& material,
   Vec2 uv,
   Vec3 geometric
) {
   if (material.normal_texture < 0 || view.tangents == nullptr || view.normals == nullptr) {
      return shading_normal(view, hit, geometric);
   }
   const std::uint32_t i0 = view.indices[(3 * hit.triangle) + 0];
   const std::uint32_t i1 = view.indices[(3 * hit.triangle) + 1];
   const std::uint32_t i2 = view.indices[(3 * hit.triangle) + 2];
   const float w0 = 1.0f - hit.u - hit.v;
   const Vec3 n = (w0 * view.normals[i0]) + (hit.u * view.normals[i1]) + (hit.v * view.normals[i2]);
   const Vec3 t = (w0 * view.tangents[i0].direction) + (hit.u * view.tangents[i1].direction) +
                  (hit.v * view.tangents[i2].direction);
   if (length_squared(n) < 1e-12f) {
      return geometric;
   }
   const Vec3 normal = normalize(n);
   // Repère tangent orthonormé autour de la normale interpolée
   const Vec3 tangent = t - (normal * dot(normal, t));
   if (length_squared(tangent) < 1e-12f) {
      return shading_normal(view, hit, geometric);
   }
   const Vec3 tx = normalize(tangent);
   const Vec3 bx = cross(normal, tx) * view.tangents[i0].sign;

   const Texture texture = view.textures[material.normal_texture];
   const Vec3 c = (sample_texture(texture, view.texels, uv) * 2.0f) - Vec3{1.0f, 1.0f, 1.0f};
   const Vec3 mapped = (tx * (c.x * material.normal_scale)) + (bx * (c.y * material.normal_scale)) +
                       (normal * c.z);
   if (length_squared(mapped) < 1e-12f) {
      return shading_normal(view, hit, geometric);
   }
   const Vec3 result = normalize(mapped);
   return dot(result, geometric) < 0.0f ? -result : result;
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

// Coordonnées de texture du point de coordonnées barycentriques (u, v) du triangle, (0, 0) si son
// matériau n'a pas de texture (une scène sans textures peut ne pas fournir de coordonnées)
RT_HOST_DEVICE inline Vec2 triangle_uv(SceneView view, std::uint32_t triangle, float u, float v) {
   const Material& material = view.materials[view.material_indices[triangle]];
   if (material.base_color_texture < 0 && material.metallic_roughness_texture < 0 &&
       material.emissive_texture < 0 && material.normal_texture < 0) {
      return {0.0f, 0.0f};
   }
   const Vec2 uv0 = view.uvs[view.indices[(3 * triangle) + 0]];
   const Vec2 uv1 = view.uvs[view.indices[(3 * triangle) + 1]];
   const Vec2 uv2 = view.uvs[view.indices[(3 * triangle) + 2]];
   return ((1.0f - u - v) * uv0) + (u * uv1) + (v * uv2);
}

// Paramètres de BSDF au point de coordonnées de texture uv, textures appliquées
RT_HOST_DEVICE inline SurfaceParams surface_params(
   SceneView view,
   const Material& material,
   Vec2 uv
) {
   SurfaceParams params{
      .base_color = material.albedo,
      .metallic = material.metallic,
      .roughness = material.roughness,
   };
   if (material.base_color_texture >= 0) {
      const Texture texture = view.textures[material.base_color_texture];
      params.base_color = params.base_color * sample_texture(texture, view.texels, uv);
   }
   if (material.metallic_roughness_texture >= 0) {
      const Texture texture = view.textures[material.metallic_roughness_texture];
      const Vec3 texel = sample_texture(texture, view.texels, uv);
      params.roughness *= texel.y;
      params.metallic *= texel.z;
   }
   return params;
}

// Lumière émise au point de coordonnées de texture uv
RT_HOST_DEVICE inline Vec3 surface_emission(SceneView view, const Material& material, Vec2 uv) {
   if (material.emissive_texture < 0) {
      return material.emission;
   }
   const Texture texture = view.textures[material.emissive_texture];
   return material.emission * sample_texture(texture, view.texels, uv);
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
   float probability,
   Pcg32& rng,
   std::uint32_t& ray_count
) {
   const Vec3 v0 = view.positions[view.indices[(3 * triangle) + 0]];
   const Vec3 v1 = view.positions[view.indices[(3 * triangle) + 1]];
   const Vec3 v2 = view.positions[view.indices[(3 * triangle) + 2]];
   const Vec2 b = sample_triangle_barycentrics(rng);
   const Vec3 y = v0 + (b.x * (v1 - v0)) + (b.y * (v2 - v0));

   const Vec3 cross_l = cross(v1 - v0, v2 - v0);
   const float area = 0.5f * length(cross_l);
   const Vec3 n_l = normalize(cross_l);

   const Vec3 x = shading.position;
   const Vec3 n = shading.frame.normal;
   const Vec3 n_g = shading.geometric_normal;
   const float d2 = length_squared(y - x);
   const Vec3 wi = (y - x) / std::sqrt(d2);

   const Material material = view.materials[view.material_indices[triangle]];
   const float cos_x = dot(n, wi);
   // Une lampe à simple face n'éclaire que les points situés du côté de sa normale
   const float cos_y = material.double_sided ? std::fabs(dot(n_l, wi)) : -dot(n_l, wi);
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

   // Densité du point tiré = probabilité du triangle / aire, ramenée en angle solide par d² / cos_y
   const float pdf_light = probability * d2 / (cos_y * area);
   const float bsdf_pdf = principled::pdf(shading.params, shading.wo, shading.frame.to_local(wi));
   const float weight = view.use_mis ? power_heuristic(pdf_light, bsdf_pdf) : 1.0f;
   const float terminator = shadow_terminator(n_g, n, wi);
   const Vec3 emission = surface_emission(view, material, triangle_uv(view, triangle, b.x, b.y));
   return f * emission * (cos_x * weight * terminator / pdf_light);
}

// Éclairage direct par une lumière ponctuelle. Sans surface, elle ne peut pas être touchée par un
// rebond : pas de MIS, toute la contribution vient d'ici
RT_HOST_DEVICE inline Vec3 direct_from_punctual(
   SceneView view,
   const ShadingPoint& shading,
   const PunctualLight& light,
   float probability,
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
   const float terminator = shadow_terminator(n_g, n, wi);
   return f * irradiance * (cos_x * terminator / probability);
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
   const LightChoice light = pick_light(view, rng.next_float());
   if (light.index < view.emissive_triangle_count) {
      const std::uint32_t triangle = view.emissive_triangles[light.index];
      return direct_from_triangle(view, shading, triangle, light.probability, rng, ray_count);
   }
   const PunctualLight& punctual = view.punctual_lights[light.index - view.emissive_triangle_count];
   return direct_from_punctual(view, shading, punctual, light.probability, ray_count);
}

// Réflexion ou réfraction sur une surface diélectrique lisse. Met à jour throughput (absorption
// du volume traversé, couleur transmise) et renvoie le rayon suivant.
// Verre dispersif : au premier passage, le chemin tire une longueur d'onde (wavelength, 0 tant
// qu'aucune n'est choisie) et prend sa couleur ; l'indice dépend ensuite de cette longueur d'onde.
RT_HOST_DEVICE inline Ray scatter_glass(
   SceneView view,
   const Hit& hit,
   const Material& material,
   Vec3 albedo,
   const Ray& ray,
   Pcg32& rng,
   Vec3& throughput,
   float& wavelength
) {
   float ior = material.ior;
   if (material.dispersion > 0.0f) {
      if (wavelength == 0.0f) {
         wavelength = sample_wavelength(rng.next_float());
         throughput = throughput * wavelength_rgb(wavelength);
      }
      ior = dispersive_ior(material.ior, material.dispersion, wavelength);
   }
   const Vec3 outward = geometric_normal(view, hit.triangle);
   if (dot(ray.direction, outward) > 0.0f) {
      // Le rayon sort de l'objet : il a traversé le volume sur une distance hit.t
      throughput = throughput * beer_lambert(material.absorption, hit.t);
   }
   const Vec3 outward_shading = shading_normal(view, hit, outward);
   const DielectricSample sample = sample_dielectric(ray.direction, outward_shading, ior, rng);
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
   const SurfaceParams& params,
   Vec2 uv,
   const Ray& ray
) {
   const Vec3 n_g = facing_normal(view, hit.triangle, ray.direction);
   // Normale d'ombrage, sauf si l'observateur se retrouve derrière elle
   const Vec3 n_s = mapped_normal(view, hit, material, uv, n_g);
   const Vec3 n = dot(n_s, -ray.direction) > 0.0f ? n_s : n_g;
   const Frame frame = make_frame(n);
   return {
      .position = ray.at(hit.t),
      .geometric_normal = n_g,
      .frame = frame,
      .wo = frame.to_local(-ray.direction),
      .params = params,
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

// État d'un chemin entre deux rebonds
struct PathState {
   Ray ray;
   Vec3 radiance{0.0f, 0.0f, 0.0f};
   Vec3 throughput{1.0f, 1.0f, 1.0f};
   int bounce = 0;
   float distance = 0.0f; // parcourue depuis la caméra
   // Après un rebond spéculaire, l'éclairage direct n'a pas été calculé : l'émission touchée compte
   // entièrement. Après un rebond de la BSDF, elle est pondérée par MIS face à l'éclairage direct.
   bool specular_bounce = false;
   float bsdf_pdf = 0.0f;
   // Avec les photons : le chemin est passé par une surface diffuse, puis seulement par du verre ou
   // du métal poli. La lumière qu'il trouve alors est une caustique, déjà apportée à cette surface
   // par les photons.
   bool after_diffuse = false;
   bool caustic_path = false;
   float wavelength = 0.0f; // nm, 0 : pas encore choisie
};

// Avance le chemin d'un rebond. Renvoie false quand il est terminé : path.radiance est alors la
// lumière qu'il rapporte. ray_count est incrémenté du nombre de rayons lancés.
RT_HOST_DEVICE inline bool trace_bounce(
   PathState& path,
   SceneView view,
   Pcg32& rng,
   std::uint32_t& ray_count,
   FirstHit* first_hit
) {
   const bool photons = view.caustics.bucket_count > 0;
   // Seul l'éclairage indirect est écrêté
   const float max_contribution = path.bounce > 0 ? view.firefly_clamp : 0.0f;
   Hit hit{};
   ++ray_count;
   if (!intersect_scene(path.ray, view, hit)) {
      if (path.bounce == 0 && first_hit != nullptr) {
         *first_hit = {.albedo = view.environment.radiance(path.ray.direction)};
      }
      const Vec3 sky = path.throughput * view.environment.radiance(path.ray.direction);
      path.radiance += clamp_contribution(sky, max_contribution);
      return false;
   }

   path.distance += hit.t;
   const Material material = view.materials[view.material_indices[hit.triangle]];
   const Vec2 uv = triangle_uv(view, hit.triangle, hit.u, hit.v);
   const bool emits_toward_ray = material.double_sided ||
                                 dot(geometric_normal(view, hit.triangle), path.ray.direction) <
                                    0.0f;
   if (max_component(material.emission) > 0.0f && emits_toward_ray && !path.caustic_path) {
      const Vec3 emission = surface_emission(view, material, uv);
      if (path.bounce == 0 || path.specular_bounce) {
         path.radiance += clamp_contribution(path.throughput * emission, max_contribution);
      } else {
         // Une lumière que l'éclairage direct ne tire jamais compte en entier
         const float pdf_light = light_pdf(view, hit.triangle, path.ray.direction, hit.t);
         const float weight = pdf_light <= 0.0f ? 1.0f
                              : view.use_mis    ? power_heuristic(path.bsdf_pdf, pdf_light)
                                                : 0.0f;
         path.radiance += clamp_contribution(path.throughput * emission * weight, max_contribution);
      }
   }
   const SurfaceParams params = surface_params(view, material, uv);
   const Vec3 albedo = params.base_color;
   if (path.bounce == 0 && first_hit != nullptr) {
      const Vec3 facing = facing_normal(view, hit.triangle, path.ray.direction);
      *first_hit = {.albedo = albedo, .normal = mapped_normal(view, hit, material, uv, facing)};
   }

   if (material.transmission > 0.0f && rng.next_float() < material.transmission) {
      // Surface diélectrique : réflexion ou réfraction, sans éclairage direct
      path.ray = scatter_glass(
         view,
         hit,
         material,
         albedo,
         path.ray,
         rng,
         path.throughput,
         path.wavelength
      );
      path.specular_bounce = true;
      path.caustic_path = path.after_diffuse;
   } else {
      // Surface opaque (glTF metallic-roughness) : éclairage direct, puis rebond selon la BSDF
      const ShadingPoint shading = make_shading_point(view, hit, material, params, uv, path.ray);
      const Vec3 n_g = shading.geometric_normal;
      const Vec3 n = shading.frame.normal;
      const bool glossy = photons && is_glossy_metal(material);
      if (!(glossy && path.after_diffuse)) {
         const Vec3 direct = path.throughput * direct_light(view, shading, rng, ray_count);
         path.radiance += clamp_contribution(direct, max_contribution);
      }
      if (photons && !glossy) {
         // Vu depuis la caméra (à travers du verre ou un miroir au plus), le rayon suit la taille
         // d'un pixel ; après un rebond diffus, le flou ne se voit plus : rayon maximal
         const PhotonMapView& map = view.caustics;
         const float radius = path.after_diffuse || map.radius_per_distance <= 0.0f
                                 ? map.radius
                                 : std::fmin(map.radius, path.distance * map.radius_per_distance);
         path.radiance += path.throughput * gather_caustics(map, shading, radius);
         path.after_diffuse = true;
      }
      path.caustic_path = glossy && path.after_diffuse;

      const BsdfSample bsdf = principled::sample(
         shading.params,
         shading.wo,
         {rng.next_float(), rng.next_float()}
      );
      const Vec3 wi = shading.frame.to_world(bsdf.wi);
      // Une direction sous la surface réelle ne peut pas repartir de ce côté
      if (bsdf.pdf <= 0.0f || dot(wi, n_g) <= 0.0f) {
         return false;
      }
      path.ray = {shading.position + (n_g * kRayOffset), wi};
      path.throughput = path.throughput * bsdf.weight * shadow_terminator(n_g, n, wi);
      path.bsdf_pdf = bsdf.pdf;
      path.specular_bounce = false;
   }

   if (path.bounce >= kRouletteStart) {
      const float p = std::fmin(max_component(path.throughput), 0.95f);
      if (rng.next_float() >= p) {
         return false;
      }
      path.throughput /= p;
   }
   ++path.bounce;
   return path.bounce < kMaxBounces;
}

// Suit un chemin à partir du rayon caméra et renvoie la lumière qu'il rapporte.
// ray_count est incrémenté du nombre de rayons lancés (rebonds et rayons d'ombre)
RT_HOST_DEVICE inline Vec3 trace_path(
   Ray ray,
   SceneView view,
   Pcg32& rng,
   std::uint32_t& ray_count,
   FirstHit* first_hit = nullptr
) {
   PathState path{.ray = ray};
   while (trace_bounce(path, view, rng, ray_count, first_hit)) {
   }
   return path.radiance;
}

} // namespace rt
