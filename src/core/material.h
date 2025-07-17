#pragma once

#include "core/vec3.h"

#include <cstdint>

namespace rt {

struct Material {
   Vec3 albedo;
   Vec3 emission;
   // Index dans Scene::textures, -1 si aucune
   std::int32_t base_color_texture = -1;
   // 0 : opaque (diffus), 1 : diélectrique transparent (verre)
   float transmission = 0.0f;
   // Indice de réfraction
   float ior = 1.5f;
   // Coefficient d'absorption du volume par unité de distance (Beer-Lambert)
   Vec3 absorption{0.0f, 0.0f, 0.0f};
};

} // namespace rt
