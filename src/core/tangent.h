#pragma once

#include "core/vec3.h"

namespace rt {

// Tangente d'un sommet, pour le normal mapping (attribut TANGENT de glTF) : direction des u
// croissants, et sens de la bitangente, qui vaut sign × (normale × tangente)
struct Tangent {
   Vec3 direction{0.0f, 0.0f, 0.0f}; // nulle : pas de repère tangent
   float sign = 1.0f;
};

} // namespace rt
