#pragma once

#include "core/host_device.h"
#include "core/vec3.h"

#include <cmath>

namespace rt {

// Exposition appliquée avant le tone mapping (1 : la radiance rendue telle quelle)
inline constexpr float kDefaultExposure = 1.0f;

// Tone mapping ACES (approximation de Stephen Hill du RRT + ODT sRGB). Entrée : couleur linéaire
// sans borne. Sortie : couleur linéaire dans [0, 1], hautes lumières compressées comme sur film.
RT_HOST_DEVICE inline Vec3 aces_tonemap(Vec3 c) {
   // Passage dans l'espace de travail ACES (sRGB -> AP1, avec l'adaptation du RRT)
   const Vec3 v{
      (0.59719f * c.x) + (0.35458f * c.y) + (0.04823f * c.z),
      (0.07600f * c.x) + (0.90834f * c.y) + (0.01566f * c.z),
      (0.02840f * c.x) + (0.13383f * c.y) + (0.83777f * c.z)
   };
   // Courbe du RRT et de l'ODT, appliquée par canal
   const auto curve = [](float x) {
      const float a = (x * (x + 0.0245786f)) - 0.000090537f;
      const float b = (x * ((0.983729f * x) + 0.4329510f)) + 0.238081f;
      return a / b;
   };
   const Vec3 t{curve(v.x), curve(v.y), curve(v.z)};
   // Retour en sRGB linéaire
   const Vec3 out{
      (1.60475f * t.x) - (0.53108f * t.y) - (0.07367f * t.z),
      (-0.10208f * t.x) + (1.10813f * t.y) - (0.00605f * t.z),
      (-0.00327f * t.x) - (0.07276f * t.y) + (1.07602f * t.z)
   };
   return min(max(out, Vec3{0.0f, 0.0f, 0.0f}), Vec3{1.0f, 1.0f, 1.0f});
}

// Encodage sRGB exact d'une composante linéaire dans [0, 1] (l'inverse de srgb_to_linear)
RT_HOST_DEVICE inline float linear_to_srgb(float c) {
   return c <= 0.0031308f ? 12.92f * c : (1.055f * std::pow(c, 1.0f / 2.4f)) - 0.055f;
}

// Couleur linéaire du rendu -> composante sRGB affichable dans [0, 1], après exposition et ACES
RT_HOST_DEVICE inline Vec3 display_color(Vec3 linear, float exposure) {
   const Vec3 mapped = aces_tonemap(linear * exposure);
   return {linear_to_srgb(mapped.x), linear_to_srgb(mapped.y), linear_to_srgb(mapped.z)};
}

} // namespace rt
