#pragma once

#include "core/host_device.h"
#include "core/rng.h"
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

// Réflexion miroir de d autour de la normale n
RT_HOST_DEVICE constexpr Vec3 reflect(Vec3 d, Vec3 n) {
   return d - (2.0f * dot(d, n) * n);
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

} // namespace rt
