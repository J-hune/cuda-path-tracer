#pragma once

#include "core/host_device.h"
#include "core/vec3.h"

#include <cmath>
#include <cstdint>

namespace rt {

// Lumière ponctuelle glTF (KHR_lights_punctual) : sans surface, un rayon ne la touche jamais.
// Elle n'est vue que par l'éclairage direct.
struct PunctualLight {
   enum class Type : std::uint8_t { Point, Spot, Directional };

   Type type = Type::Point;
   Vec3 position{0.0f, 0.0f, 0.0f};
   // Direction dans laquelle la lumière éclaire (spot, directionnelle)
   Vec3 direction{0.0f, 0.0f, -1.0f};
   // Point, spot : intensité en W/sr. Directionnelle : éclairement en W/m²
   Vec3 intensity{1.0f, 1.0f, 1.0f};
   // Spot : cosinus des angles intérieur (pleine lumière) et extérieur (plus de lumière)
   float cos_inner = 1.0f;
   float cos_outer = 0.0f;
};

// Atténuation angulaire d'un spot glTF : 1 dans le cône intérieur, 0 hors du cône extérieur,
// transition douce (interpolation linéaire en cosinus, puis élevée au carré)
RT_HOST_DEVICE inline float spot_attenuation(const PunctualLight& light, Vec3 to_point) {
   const float cos_theta = dot(light.direction, to_point);
   const float range = std::fmax(light.cos_inner - light.cos_outer, 1e-4f);
   const float t = std::fmin(std::fmax((cos_theta - light.cos_outer) / range, 0.0f), 1.0f);
   return t * t;
}

} // namespace rt
