#pragma once

#include "core/host_device.h"
#include "core/vec2.h"
#include "core/vec3.h"

#include <cmath>

// Distribution de microfacettes GGX (Trowbridge-Reitz), dans le repère local (normale = +z).
// alpha : rugosité de la distribution, alpha = roughness² dans la convention glTF.
namespace rt::ggx {

// D(h) : densité des normales de microfacettes, normalisée par ∫ D(h) cos θh dωh = 1
RT_HOST_DEVICE inline float distribution(Vec3 h, float alpha) {
   if (h.z <= 0.0f) {
      return 0.0f;
   }
   const float a2 = alpha * alpha;
   const float d = (h.z * h.z * (a2 - 1.0f)) + 1.0f;
   return a2 / (kPi * d * d);
}

// Λ(w) de Smith pour GGX : part des microfacettes cachées vue depuis la direction w
RT_HOST_DEVICE inline float lambda(Vec3 w, float alpha) {
   const float z2 = w.z * w.z;
   if (z2 <= 0.0f) {
      return kInfinity;
   }
   const float tan2 = ((w.x * w.x) + (w.y * w.y)) / z2;
   return 0.5f * (-1.0f + std::sqrt(1.0f + (alpha * alpha * tan2)));
}

// Masquage d'une direction
RT_HOST_DEVICE inline float smith_g1(Vec3 w, float alpha) {
   return 1.0f / (1.0f + lambda(w, alpha));
}

// Masquage-ombrage corrélé en hauteur : une microfacette cachée pour wo l'est souvent aussi pour wi
RT_HOST_DEVICE inline float smith_g2(Vec3 wo, Vec3 wi, float alpha) {
   return 1.0f / (1.0f + lambda(wo, alpha) + lambda(wi, alpha));
}

// Tire une normale de microfacette visible depuis wo (wo.z > 0), de densité
// G1(wo) · max(0, wo·h) · D(h) / wo.z (Dupuy et Benyoub, « Sampling Visible GGX Normals with
// Spherical Caps », 2023)
RT_HOST_DEVICE inline Vec3 sample_visible_normal(Vec3 wo, float alpha, Vec2 u) {
   // Passage dans l'espace où la distribution devient une hémisphère (« stretch »)
   const Vec3 wo_std = normalize({wo.x * alpha, wo.y * alpha, wo.z});
   // Tirage uniforme sur la calotte sphérique visible depuis wo_std
   const float phi = 2.0f * kPi * u.x;
   const float z = ((1.0f - u.y) * (1.0f + wo_std.z)) - wo_std.z;
   const float sin_theta = std::sqrt(std::fmax(0.0f, 1.0f - (z * z)));
   const Vec3 c{sin_theta * std::cos(phi), sin_theta * std::sin(phi), z};
   const Vec3 h_std = c + wo_std;
   // Retour dans l'espace de la distribution (« unstretch »)
   return normalize({h_std.x * alpha, h_std.y * alpha, std::fmax(h_std.z, 0.0f)});
}

// Densité, par unité d'angle solide de wi, d'une direction réfléchie autour d'une normale visible
RT_HOST_DEVICE inline float reflection_pdf(Vec3 wo, Vec3 wi, float alpha) {
   if (wo.z <= 0.0f || wi.z <= 0.0f) {
      return 0.0f;
   }
   const Vec3 h = normalize(wo + wi);
   return smith_g1(wo, alpha) * distribution(h, alpha) / (4.0f * wo.z);
}

} // namespace rt::ggx
