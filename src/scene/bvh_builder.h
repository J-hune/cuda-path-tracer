#pragma once

#include "core/bvh.h"
#include "scene/gltf_loader.h"

#include <cstdint>
#include <vector>

namespace rt {

struct Bvh {
   std::vector<BvhNode> nodes;                  // nodes[0] est la racine
   std::vector<std::uint32_t> triangle_indices; // indices de triangles, réordonnés par feuille
};

Bvh build_bvh(const Scene& scene);

} // namespace rt
