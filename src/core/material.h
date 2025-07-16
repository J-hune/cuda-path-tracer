#pragma once

#include "core/vec3.h"

#include <cstdint>

namespace rt {

struct Material {
   Vec3 albedo;
   Vec3 emission;
   std::int32_t base_color_texture = -1; // index dans Scene::textures, -1 si aucune
};

} // namespace rt
