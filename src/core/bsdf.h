#pragma once

#include "core/host_device.h"
#include "core/sampling.h"
#include "core/vec2.h"
#include "core/vec3.h"

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

} // namespace rt
