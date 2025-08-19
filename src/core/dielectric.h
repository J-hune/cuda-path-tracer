#pragma once

#include "core/bsdf.h"
#include "core/ggx.h"
#include "core/host_device.h"
#include "core/rng.h"
#include "core/vec2.h"
#include "core/vec3.h"

#include <cmath>

namespace rt {

// Part de la lumière réfléchie par une interface (approximation de Schlick).
// cos_theta : cosinus de l'angle du côté du milieu le moins dense (l'air)
RT_HOST_DEVICE inline float fresnel_schlick(float cos_theta, float ior) {
   const float r = (1.0f - ior) / (1.0f + ior);
   const float r0 = r * r;
   const float m = 1.0f - cos_theta;
   return r0 + ((1.0f - r0) * m * m * m * m * m);
}

// Atténuation de Beer-Lambert après une distance parcourue dans un milieu absorbant
RT_HOST_DEVICE inline Vec3 beer_lambert(Vec3 absorption, float distance) {
   return {
      std::exp(-absorption.x * distance),
      std::exp(-absorption.y * distance),
      std::exp(-absorption.z * distance)
   };
}

struct DielectricSample {
   Vec3 direction;
   bool transmitted = false; // true : réfracté (traverse la surface), false : réfléchi
};

// Diffusion sur une interface diélectrique lisse (verre, eau).
// d : direction incidente unitaire ; outward : normale géométrique vers l'extérieur de l'objet.
// Réflexion avec la probabilité de Fresnel, sinon réfraction (loi de Snell).
// Chaque choix a un poids de 1 : l'énergie est conservée.
RT_HOST_DEVICE inline DielectricSample sample_dielectric(
   Vec3 d,
   Vec3 outward,
   float ior,
   Pcg32& rng
) {
   const bool entering = dot(d, outward) < 0.0f;
   const Vec3 n = entering ? outward : -outward;
   const float eta = entering ? 1.0f / ior : ior;

   const float cos_i = -dot(d, n);
   const float sin2_t = eta * eta * (1.0f - (cos_i * cos_i));
   if (sin2_t >= 1.0f) {
      return {.direction = reflect(d, n), .transmitted = false}; // réflexion totale interne
   }

   const float cos_t = std::sqrt(1.0f - sin2_t);
   const float fresnel = fresnel_schlick(entering ? cos_i : cos_t, ior);
   if (rng.next_float() < fresnel) {
      return {.direction = reflect(d, n), .transmitted = false};
   }

   const Vec3 refracted = (eta * d) + (((eta * cos_i) - cos_t) * n);
   return {.direction = normalize(refracted), .transmitted = true};
}

// Verre dépoli : microfacettes GGX qui réfléchissent ou réfractent chacune comme une interface
// lisse (Walter et al., « Microfacet Models for Refraction through Rough Surfaces », 2007). wo est
// du côté de la normale (wo.z > 0) ; eta : indice du milieu opposé sur celui de wo. La BSDF ne
// compte qu'une réflexion ou réfraction par microfacette.
namespace rough_dielectric {

// Part réfléchie par une microfacette vue sous le cosinus cos_o depuis wo
RT_HOST_DEVICE inline float fresnel(float cos_o, float eta) {
   if (eta >= 1.0f) {
      return fresnel_schlick(cos_o, eta);
   }
   const float sin2_t = (1.0f - (cos_o * cos_o)) / (eta * eta);
   if (sin2_t >= 1.0f) {
      return 1.0f;
   }
   return fresnel_schlick(std::sqrt(1.0f - sin2_t), 1.0f / eta);
}

// Normale de microfacette qui envoie wo vers wi (demi-vecteur généralisé), nulle si aucune
RT_HOST_DEVICE inline Vec3 half_vector(Vec3 wo, Vec3 wi, float eta) {
   const Vec3 h = wi.z > 0.0f ? wo + wi : wo + (wi * eta);
   if (length_squared(h) <= 0.0f) {
      return {0.0f, 0.0f, 0.0f};
   }
   const Vec3 m = normalize(h);
   const Vec3 up = m.z < 0.0f ? -m : m;
   // La microfacette doit faire face à wo et à wi
   if (dot(wo, up) <= 0.0f || dot(wi, up) * wi.z <= 0.0f) {
      return {0.0f, 0.0f, 0.0f};
   }
   return up;
}

RT_HOST_DEVICE inline float eval(Vec3 wo, Vec3 wi, float alpha, float eta) {
   if (wo.z <= 0.0f || wi.z == 0.0f) {
      return 0.0f;
   }
   const Vec3 m = half_vector(wo, wi, eta);
   if (m.z <= 0.0f) {
      return 0.0f;
   }
   const float cos_om = dot(wo, m);
   const float cos_im = dot(wi, m);
   const float f = fresnel(cos_om, eta);
   const float dg = ggx::distribution(m, alpha) * ggx::smith_g2(wo, wi, alpha);
   if (wi.z > 0.0f) {
      return f * dg / (4.0f * wo.z * wi.z);
   }
   const float denom = cos_im + (cos_om / eta);
   return (1.0f - f) * dg * std::fabs(cos_im * cos_om / (wo.z * wi.z * denom * denom));
}

RT_HOST_DEVICE inline float pdf(Vec3 wo, Vec3 wi, float alpha, float eta) {
   if (wo.z <= 0.0f || wi.z == 0.0f) {
      return 0.0f;
   }
   const Vec3 m = half_vector(wo, wi, eta);
   if (m.z <= 0.0f) {
      return 0.0f;
   }
   const float cos_om = dot(wo, m);
   const float f = fresnel(cos_om, eta);
   const float visible = ggx::smith_g1(wo, alpha) * ggx::distribution(m, alpha) * cos_om / wo.z;
   if (wi.z > 0.0f) {
      return f * visible / (4.0f * cos_om);
   }
   const float cos_im = dot(wi, m);
   const float denom = cos_im + (cos_om / eta);
   return (1.0f - f) * visible * std::fabs(cos_im) / (denom * denom);
}

// u choisit la microfacette visible, u_lobe la réflexion ou la réfraction
RT_HOST_DEVICE inline BsdfSample sample(Vec3 wo, float alpha, float eta, Vec2 u, float u_lobe) {
   const BsdfSample failed{.wi = {0.0f, 0.0f, 1.0f}, .weight = {0.0f, 0.0f, 0.0f}, .pdf = 0.0f};
   if (wo.z <= 0.0f) {
      return failed;
   }
   const Vec3 m = ggx::sample_visible_normal(wo, alpha, u);
   const float cos_om = dot(wo, m);
   Vec3 wi{0.0f, 0.0f, 1.0f};
   const bool reflected = u_lobe < fresnel(cos_om, eta);
   if (reflected) {
      wi = reflect(-wo, m);
   } else {
      const float sin2_t = (1.0f - (cos_om * cos_om)) / (eta * eta);
      if (sin2_t >= 1.0f) {
         return failed;
      }
      const float cos_t = std::sqrt(1.0f - sin2_t);
      wi = normalize((-wo / eta) + (((cos_om / eta) - cos_t) * m));
   }
   // Direction passée du mauvais côté de la surface moyenne
   if ((wi.z > 0.0f) != reflected) {
      return failed;
   }
   const float density = pdf(wo, wi, alpha, eta);
   if (density <= 0.0f) {
      return failed;
   }
   const float weight = eval(wo, wi, alpha, eta) * std::fabs(wi.z) / density;
   return {.wi = wi, .weight = {weight, weight, weight}, .pdf = density};
}

} // namespace rough_dielectric

} // namespace rt
