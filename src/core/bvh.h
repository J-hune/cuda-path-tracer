#pragma once

#include "core/aabb.h"
#include "core/vec3.h"

#include <cstdint>

namespace rt {

// Sommets d'un triangle, rangés dans l'ordre des feuilles : une seule lecture contiguë par triangle
// pendant le parcours
struct BvhTriangle {
   Vec3 v0;
   Vec3 v1;
   Vec3 v2;
};

// Nœud de BVH, stocké dans un tableau plat (copiable tel quel sur le GPU).
// Feuille : count > 0, triangles [first, first + count) de triangle_indices et de triangles.
// Nœud interne : count == 0, enfants en first (gauche) et first + 1 (droite).
struct BvhNode {
   Aabb bounds;
   std::uint32_t first = 0;
   std::uint32_t count = 0;
};

} // namespace rt
