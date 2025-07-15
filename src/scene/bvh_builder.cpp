#include "scene/bvh_builder.h"

#include "core/aabb.h"
#include "core/bvh.h"
#include "core/vec3.h"
#include "scene/gltf_loader.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace rt {
namespace {

constexpr std::uint32_t kMaxLeafSize = 2;

struct BuildContext {
   const Scene& scene;
   std::vector<Vec3> centroids; // centre de chaque triangle
   Bvh bvh;
};

constexpr float component(Vec3 v, int axis) {
   return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

Vec3 vertex(const Scene& scene, std::uint32_t triangle, std::uint32_t corner) {
   return scene.positions[scene.indices[(3 * static_cast<std::size_t>(triangle)) + corner]];
}

// Calcule la boîte englobante des triangles du nœud
void update_bounds(BuildContext& ctx, std::uint32_t node_index) {
   BvhNode& node = ctx.bvh.nodes[node_index];
   node.bounds = Aabb{};
   for (std::uint32_t i = node.first; i < node.first + node.count; ++i) {
      const std::uint32_t triangle = ctx.bvh.triangle_indices[i];
      for (std::uint32_t corner = 0; corner < 3; ++corner) {
         node.bounds.grow(vertex(ctx.scene, triangle, corner));
      }
   }
}

// Coupe le nœud en deux enfants si c'est utile, puis recommence sur chacun
void subdivide(BuildContext& ctx, std::uint32_t node_index) {
   const std::uint32_t first = ctx.bvh.nodes[node_index].first;
   const std::uint32_t count = ctx.bvh.nodes[node_index].count;
   if (count <= kMaxLeafSize) {
      return;
   }

   // Coupe au milieu de l'axe le plus long de la boîte des centres
   Aabb centroid_bounds;
   for (std::uint32_t i = first; i < first + count; ++i) {
      centroid_bounds.grow(ctx.centroids[ctx.bvh.triangle_indices[i]]);
   }
   const Vec3 extent = centroid_bounds.max - centroid_bounds.min;
   int axis = 0;
   if (extent.y > extent.x) {
      axis = 1;
   }
   if (extent.z > component(extent, axis)) {
      axis = 2;
   }
   const float split = component(centroid_bounds.min, axis) + (0.5f * component(extent, axis));

   const auto begin = ctx.bvh.triangle_indices.begin() + first;
   const auto middle = std::partition(begin, begin + count, [&](std::uint32_t triangle) {
      return component(ctx.centroids[triangle], axis) < split;
   });
   const auto left_count = static_cast<std::uint32_t>(middle - begin);
   if (left_count == 0 || left_count == count) {
      return;
   }

   const auto left_index = static_cast<std::uint32_t>(ctx.bvh.nodes.size());
   ctx.bvh.nodes.push_back({.bounds = {}, .first = first, .count = left_count});
   ctx.bvh.nodes.push_back(
      {.bounds = {}, .first = first + left_count, .count = count - left_count}
   );
   ctx.bvh.nodes[node_index].first = left_index;
   ctx.bvh.nodes[node_index].count = 0;

   update_bounds(ctx, left_index);
   update_bounds(ctx, left_index + 1);
   subdivide(ctx, left_index);
   subdivide(ctx, left_index + 1);
}

} // namespace

Bvh build_bvh(const Scene& scene) {
   const auto triangle_count = static_cast<std::uint32_t>(scene.indices.size() / 3);
   BuildContext ctx{.scene = scene, .centroids = {}, .bvh = {}};
   if (triangle_count == 0) {
      return ctx.bvh;
   }

   ctx.centroids.reserve(triangle_count);
   for (std::uint32_t triangle = 0; triangle < triangle_count; ++triangle) {
      const Vec3 sum = vertex(scene, triangle, 0) + vertex(scene, triangle, 1) +
                       vertex(scene, triangle, 2);
      ctx.centroids.push_back(sum / 3.f);
   }

   ctx.bvh.triangle_indices.resize(triangle_count);
   std::ranges::iota(ctx.bvh.triangle_indices, 0u);

   ctx.bvh.nodes.reserve((2 * static_cast<std::size_t>(triangle_count)) - 1);
   ctx.bvh.nodes.push_back({.bounds = {}, .first = 0, .count = triangle_count});
   update_bounds(ctx, 0);
   subdivide(ctx, 0);

   return ctx.bvh;
}

} // namespace rt
