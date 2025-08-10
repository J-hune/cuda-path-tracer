#pragma once

#include "core/bvh.h"
#include "scene/gltf_loader.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rt {

struct Bvh {
   std::vector<BvhNode> nodes;                  // nodes[0] est la racine
   std::vector<std::uint32_t> triangle_indices; // indices de triangles, réordonnés par feuille
   std::vector<BvhTriangle> triangles;          // leurs sommets, dans le même ordre
};

// Stratégie de découpe d'un nœud en deux enfants
enum class BvhSplit : std::uint8_t {
   Median, // milieu de l'axe le plus long (simple, sert de référence)
   Sah,    // Surface Area Heuristic par binning : minimise le coût attendu d'un rayon
};

Bvh build_bvh(const Scene& scene, BvhSplit split = BvhSplit::Sah);

// Statistiques de qualité d'un BVH
struct BvhStats {
   std::size_t node_count = 0;
   std::size_t leaf_count = 0;
   double average_depth = 0.0; // profondeur moyenne des feuilles
   std::size_t max_depth = 0;
   double average_leaf_size = 0.0; // triangles par feuille
};

BvhStats compute_stats(const Bvh& bvh);

} // namespace rt
