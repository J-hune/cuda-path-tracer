#pragma once

#include "core/ggx.h"
#include "core/host_device.h"
#include "core/sampling.h"
#include "core/vec2.h"
#include "core/vec3.h"

#include <cmath>

namespace rt {

// Interface commune des BSDF. Toutes les directions sont dans le repère local de la surface
// (normale = +z) et pointent vers l'extérieur du point : wo vers l'observateur, wi vers la lumière.
//   eval(wo, wi)   : valeur f de la BSDF (sans le cosinus)
//   pdf(wo, wi)    : densité de probabilité de tirer wi, par unité d'angle solide
//   sample(wo, u)  : tire wi à partir de deux nombres uniformes u ∈ [0, 1]²
// Les trois doivent être cohérentes : sample tire selon pdf, et weight = eval · cos θi / pdf.
struct BsdfSample {
   Vec3 wi;
   Vec3 weight; // f · cos θi / pdf : ce par quoi le throughput est multiplié
   float pdf = 0.0f;
};

// Surface parfaitement mate : renvoie la lumière uniformément, f = albedo / π
namespace lambert {

RT_HOST_DEVICE constexpr Vec3 eval(Vec3 albedo, Vec3 wo, Vec3 wi) {
   if (wo.z <= 0.0f || wi.z <= 0.0f) {
      return {0.0f, 0.0f, 0.0f};
   }
   return albedo / kPi;
}

RT_HOST_DEVICE constexpr float pdf(Vec3 wo, Vec3 wi) {
   return wo.z > 0.0f ? cosine_hemisphere_pdf(wi) : 0.0f;
}

// Tirage en cosinus : f · cos / pdf = albedo, tous les autres termes s'annulent
RT_HOST_DEVICE inline BsdfSample sample(Vec3 albedo, Vec3 wo, Vec2 u) {
   const Vec3 wi = square_to_cosine_hemisphere(u);
   const float density = pdf(wo, wi);
   if (density <= 0.0f) {
      return {.wi = wi, .weight = {0.0f, 0.0f, 0.0f}, .pdf = 0.0f};
   }
   return {.wi = wi, .weight = albedo, .pdf = density};
}

} // namespace lambert

// Fresnel de Schlick avec une réflectance à incidence normale f0 par canal (métaux colorés)
RT_HOST_DEVICE inline Vec3 schlick(Vec3 f0, float cos_theta) {
   const float m = std::fmax(1.0f - cos_theta, 0.0f);
   const float m5 = m * m * m * m * m;
   return f0 + ((Vec3{1.0f, 1.0f, 1.0f} - f0) * m5);
}

// Paramètres du modèle metallic-roughness de glTF au point touché
struct SurfaceParams {
   Vec3 base_color{1.0f, 1.0f, 1.0f};
   float metallic = 0.0f;  // 0 : diélectrique (diffus + reflet blanc), 1 : métal (reflet coloré)
   float roughness = 1.0f; // 0 : lisse, 1 : très rugueux
};

// BSDF metallic-roughness de glTF : mélange, selon metallic, de
//  - un métal : microfacettes GGX, réflectance de Schlick colorée par base_color ;
//  - un diélectrique : diffus lambertien de couleur base_color sous un reflet GGX de 4 %.
// Modèle à une seule réflexion : un métal très rugueux perd de l'énergie (la lumière qui rebondit
// entre plusieurs microfacettes n'est pas comptée).
namespace principled {

inline constexpr float kMinAlpha = 1e-3f;     // rugosité minimale, évite une distribution de Dirac
inline constexpr float kDielectricF0 = 0.04f; // réflectance à incidence normale d'un IOR de 1,5

RT_HOST_DEVICE inline float alpha(float roughness) {
   return std::fmax(roughness * roughness, kMinAlpha);
}

// Probabilité de tirer le reflet plutôt que le diffus
RT_HOST_DEVICE inline float specular_probability(const SurfaceParams& params) {
   return params.metallic + ((1.0f - params.metallic) * 0.25f);
}

RT_HOST_DEVICE inline Vec3 eval(const SurfaceParams& params, Vec3 wo, Vec3 wi) {
   if (wo.z <= 0.0f || wi.z <= 0.0f) {
      return {0.0f, 0.0f, 0.0f};
   }
   const float a = alpha(params.roughness);
   const Vec3 h = normalize(wo + wi);
   const float cos_d = std::fmax(dot(wo, h), 0.0f);
   const float specular = ggx::distribution(h, a) * ggx::smith_g2(wo, wi, a) / (4.0f * wo.z * wi.z);

   const Vec3 metal = schlick(params.base_color, cos_d) * specular;

   // Diélectrique : la lumière réfléchie par le reflet n'entre pas dans la couche diffuse.
   // Le diffus est pondéré par la part transmise à l'entrée et à la sortie, (1 - F(θo))(1 - F(θi))
   // : l'énergie est conservée même en incidence rasante, et la BSDF reste réciproque.
   const Vec3 f0{kDielectricF0, kDielectricF0, kDielectricF0};
   const float fresnel = schlick(f0, cos_d).x;
   const float transmitted = (1.0f - schlick(f0, wo.z).x) * (1.0f - schlick(f0, wi.z).x);
   const Vec3 dielectric = (params.base_color * (transmitted / kPi)) +
                           Vec3{fresnel * specular, fresnel * specular, fresnel * specular};
   return (dielectric * (1.0f - params.metallic)) + (metal * params.metallic);
}

RT_HOST_DEVICE inline float pdf(const SurfaceParams& params, Vec3 wo, Vec3 wi) {
   if (wo.z <= 0.0f || wi.z <= 0.0f) {
      return 0.0f;
   }
   const float p_specular = specular_probability(params);
   return (p_specular * ggx::reflection_pdf(wo, wi, alpha(params.roughness))) +
          ((1.0f - p_specular) * cosine_hemisphere_pdf(wi));
}

// u.x choisit d'abord le lobe, puis est réutilisé (remis dans [0, 1[) pour tirer la direction
RT_HOST_DEVICE inline BsdfSample sample(const SurfaceParams& params, Vec3 wo, Vec2 u) {
   const BsdfSample failed{.wi = {0.0f, 0.0f, 1.0f}, .weight = {0.0f, 0.0f, 0.0f}, .pdf = 0.0f};
   if (wo.z <= 0.0f) {
      return failed;
   }

   const float p_specular = specular_probability(params);
   Vec3 wi{0.0f, 0.0f, 1.0f};
   if (u.x < p_specular) {
      const Vec2 v{u.x / p_specular, u.y};
      const Vec3 h = ggx::sample_visible_normal(wo, alpha(params.roughness), v);
      wi = reflect(-wo, h);
   } else {
      const Vec2 v{(u.x - p_specular) / (1.0f - p_specular), u.y};
      wi = square_to_cosine_hemisphere(v);
   }

   const float density = pdf(params, wo, wi);
   if (density <= 0.0f) {
      return failed;
   }
   return {.wi = wi, .weight = eval(params, wo, wi) * (wi.z / density), .pdf = density};
}

} // namespace principled

} // namespace rt
