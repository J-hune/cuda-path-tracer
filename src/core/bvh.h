#pragma once

#include "core/aabb.h"

#include <cstdint>

namespace rt {

// Nœud de BVH, stocké dans un tableau plat (copiable tel quel sur le GPU).
// Feuille : count > 0, triangles [first, first + count) de triangle_indices.
// Nœud interne : count == 0, enfants en first (gauche) et first + 1 (droite).
struct BvhNode {
   Aabb bounds;
   std::uint32_t first = 0;
   std::uint32_t count = 0;
};

} // namespace rt
