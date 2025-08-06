#pragma once

#include "core/host_device.h"
#include "core/vec3.h"

#include <cmath>
#include <cstdint>

namespace rt {

// Standard : aucune courbe, la couleur est seulement bornée à [0, 1]
enum class ToneMapping : std::uint8_t { Agx, Aces, Standard };

// Passage de la radiance rendue à l'écran, comme dans la gestion des couleurs de Blender
struct DisplaySettings {
   float exposure = 0.0f; // en IL (EV) : la radiance est multipliée par 2^exposure
   ToneMapping tone_mapping = ToneMapping::Agx;
};

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

// AgX, la transformation de vue par défaut de Blender 4. Approximation publiée par Benjamin Wrensch
// (« Minimal AgX Implementation », 2023) : courbe polynomiale ajustée sur AgX Base. Entrée :
// couleur linéaire sans borne. Sortie : couleur linéaire dans [0, 1], hautes lumières désaturées.
RT_HOST_DEVICE inline Vec3 agx_tonemap(Vec3 c) {
   // Passage dans l'espace de travail d'AgX (primaires resserrées)
   const Vec3 v{
      (0.842479062f * c.x) + (0.078433600f * c.y) + (0.079223745f * c.z),
      (0.042328242f * c.x) + (0.878468636f * c.y) + (0.079166127f * c.z),
      (0.042375655f * c.x) + (0.078433600f * c.y) + (0.879142974f * c.z)
   };
   // Logarithme normalisé sur la plage [-12,47 ; 4,03] IL, puis courbe de contraste
   const auto curve = [](float x) {
      constexpr float kMinEv = -12.47393f;
      constexpr float kMaxEv = 4.026069f;
      const float log = std::fmin(std::fmax(std::log2(std::fmax(x, 1e-10f)), kMinEv), kMaxEv);
      const float t = (log - kMinEv) / (kMaxEv - kMinEv);
      const float t2 = t * t;
      const float t4 = t2 * t2;
      return (15.5f * t4 * t2) - (40.14f * t4 * t) + (31.96f * t4) - (6.868f * t2 * t) +
             (0.4298f * t2) + (0.1191f * t) - 0.00232f;
   };
   const Vec3 t{curve(v.x), curve(v.y), curve(v.z)};
   // Retour aux primaires sRGB, puis en linéaire
   const Vec3 out{
      (1.196879005f * t.x) - (0.098020881f * t.y) - (0.099029744f * t.z),
      (-0.052896852f * t.x) + (1.151903130f * t.y) - (0.098961177f * t.z),
      (-0.052971636f * t.x) - (0.098043450f * t.y) + (1.151073673f * t.z)
   };
   const auto to_linear = [](float x) { return std::pow(std::fmax(x, 0.0f), 2.2f); };
   return min(Vec3{to_linear(out.x), to_linear(out.y), to_linear(out.z)}, Vec3{1.0f, 1.0f, 1.0f});
}

// Encodage sRGB exact d'une composante linéaire dans [0, 1] (l'inverse de srgb_to_linear)
RT_HOST_DEVICE inline float linear_to_srgb(float c) {
   return c <= 0.0031308f ? 12.92f * c : (1.055f * std::pow(c, 1.0f / 2.4f)) - 0.055f;
}

// Couleur linéaire du rendu -> composante sRGB affichable dans [0, 1]
RT_HOST_DEVICE inline Vec3 display_color(Vec3 linear, DisplaySettings settings) {
   const Vec3 exposed = linear * std::exp2(settings.exposure);
   Vec3 mapped = min(max(exposed, Vec3{0.0f, 0.0f, 0.0f}), Vec3{1.0f, 1.0f, 1.0f});
   if (settings.tone_mapping == ToneMapping::Agx) {
      mapped = agx_tonemap(exposed);
   } else if (settings.tone_mapping == ToneMapping::Aces) {
      mapped = aces_tonemap(exposed);
   }
   return {linear_to_srgb(mapped.x), linear_to_srgb(mapped.y), linear_to_srgb(mapped.z)};
}

} // namespace rt
