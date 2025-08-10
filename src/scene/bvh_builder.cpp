#include "scene/bvh_builder.h"

#include "core/aabb.h"
#include "core/bvh.h"
#include "core/vec3.h"
#include "scene/gltf_loader.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

namespace rt {
namespace {

constexpr std::uint32_t kMedianLeafSize = 2; // découpage médian : feuilles de 2 triangles au plus
constexpr std::uint32_t kMaxLeafSize = 8;    // SAH : une feuille plus grande est toujours coupée
constexpr int kBinCount = 16;
// Coûts relatifs de la SAH : traverser un nœud, tester un triangle
constexpr float kTraversalCost = 1.0f;
constexpr float kIntersectionCost = 1.0f;

struct BuildContext {
   const Scene& scene;
   BvhSplit split;
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

// Demi-surface d'une boîte : seule la proportion entre surfaces compte pour la SAH
float half_area(const Aabb& box) {
   const Vec3 e = box.max - box.min;
   return (e.x * e.y) + (e.y * e.z) + (e.z * e.x);
}

struct Split {
   int axis = -1; // -1 : pas de coupe, le nœud reste une feuille
   float position = 0.0f;
};

Aabb triangle_bounds(const Scene& scene, std::uint32_t triangle) {
   Aabb box;
   for (std::uint32_t corner = 0; corner < 3; ++corner) {
      box.grow(vertex(scene, triangle, corner));
   }
   return box;
}

// Découpage médian : milieu de l'axe le plus long de la boîte des centres
Split find_median_split(const BvhNode& node, const Aabb& centroid_bounds) {
   if (node.count <= kMedianLeafSize) {
      return {};
   }
   const Vec3 extent = centroid_bounds.max - centroid_bounds.min;
   int axis = 0;
   if (extent.y > extent.x) {
      axis = 1;
   }
   if (extent.z > component(extent, axis)) {
      axis = 2;
   }
   return {
      .axis = axis,
      .position = component(centroid_bounds.min, axis) + (0.5f * component(extent, axis))
   };
}

// SAH par binning : les centres sont rangés dans kBinCount cases par axe, et chaque frontière entre
// cases est évaluée comme coupe. Renvoie la coupe la moins chère, ou aucune si une feuille coûte
// moins
Split find_sah_split(const BuildContext& ctx, const BvhNode& node, const Aabb& centroid_bounds) {
   struct Bin {
      Aabb bounds;
      std::uint32_t count = 0;
   };

   Split best;
   float best_cost = kIntersectionCost * static_cast<float>(node.count); // coût d'une feuille
   const float parent_area = half_area(node.bounds);
   if (node.count > kMaxLeafSize) {
      best_cost = kInfinity; // trop de triangles pour une feuille : il faut couper
   }

   for (int axis = 0; axis < 3; ++axis) {
      const float min = component(centroid_bounds.min, axis);
      const float extent = component(centroid_bounds.max, axis) - min;
      if (extent <= 0.0f) {
         continue; // tous les centres sont au même endroit sur cet axe
      }
      const float scale = static_cast<float>(kBinCount) / extent;

      std::array<Bin, kBinCount> bins{};
      for (std::uint32_t i = node.first; i < node.first + node.count; ++i) {
         const std::uint32_t triangle = ctx.bvh.triangle_indices[i];
         const float position = component(ctx.centroids[triangle], axis);
         const int b = std::min(static_cast<int>((position - min) * scale), kBinCount - 1);
         Bin& bin = bins[static_cast<std::size_t>(b)];
         ++bin.count;
         const Aabb box = triangle_bounds(ctx.scene, triangle);
         bin.bounds.grow(box.min);
         bin.bounds.grow(box.max);
      }

      // Balayage : aires et nombres de triangles à gauche et à droite de chaque frontière
      std::array<float, kBinCount - 1> left_area{};
      std::array<std::uint32_t, kBinCount - 1> left_count{};
      Aabb left;
      std::uint32_t left_sum = 0;
      for (std::size_t b = 0; b + 1 < kBinCount; ++b) {
         left_sum += bins[b].count;
         if (bins[b].count > 0) {
            left.grow(bins[b].bounds.min);
            left.grow(bins[b].bounds.max);
         }
         left_area[b] = half_area(left);
         left_count[b] = left_sum;
      }
      Aabb right;
      std::uint32_t right_sum = 0;
      for (std::size_t b = kBinCount - 1; b > 0; --b) {
         right_sum += bins[b].count;
         if (bins[b].count > 0) {
            right.grow(bins[b].bounds.min);
            right.grow(bins[b].bounds.max);
         }
         const std::size_t boundary = b - 1;
         if (left_count[boundary] == 0 || right_sum == 0) {
            continue;
         }
         const float cost = kTraversalCost +
                            (kIntersectionCost *
                             ((left_area[boundary] * static_cast<float>(left_count[boundary])) +
                              (half_area(right) * static_cast<float>(right_sum))) /
                             parent_area);
         if (cost < best_cost) {
            best_cost = cost;
            best = {.axis = axis, .position = min + (static_cast<float>(b) / scale)};
         }
      }
   }
   return best;
}

// Coupe le nœud en deux enfants si c'est utile, puis recommence sur chacun
void subdivide(BuildContext& ctx, std::uint32_t node_index) {
   const BvhNode node = ctx.bvh.nodes[node_index];
   if (node.count <= 1) {
      return;
   }

   Aabb centroid_bounds;
   for (std::uint32_t i = node.first; i < node.first + node.count; ++i) {
      centroid_bounds.grow(ctx.centroids[ctx.bvh.triangle_indices[i]]);
   }
   const Split split = ctx.split == BvhSplit::Sah ? find_sah_split(ctx, node, centroid_bounds)
                                                  : find_median_split(node, centroid_bounds);
   if (split.axis < 0) {
      return;
   }

   const auto begin = ctx.bvh.triangle_indices.begin() + node.first;
   const auto middle = std::partition(begin, begin + node.count, [&](std::uint32_t triangle) {
      return component(ctx.centroids[triangle], split.axis) < split.position;
   });
   const auto left_count = static_cast<std::uint32_t>(middle - begin);
   if (left_count == 0 || left_count == node.count) {
      return;
   }

   const auto left_index = static_cast<std::uint32_t>(ctx.bvh.nodes.size());
   ctx.bvh.nodes.push_back({.bounds = {}, .first = node.first, .count = left_count});
   ctx.bvh.nodes.push_back(
      {.bounds = {}, .first = node.first + left_count, .count = node.count - left_count}
   );
   ctx.bvh.nodes[node_index].first = left_index;
   ctx.bvh.nodes[node_index].count = 0;

   update_bounds(ctx, left_index);
   update_bounds(ctx, left_index + 1);
   subdivide(ctx, left_index);
   subdivide(ctx, left_index + 1);
}

} // namespace

Bvh build_bvh(const Scene& scene, BvhSplit split) {
   const auto triangle_count = static_cast<std::uint32_t>(scene.indices.size() / 3);
   BuildContext ctx{.scene = scene, .split = split, .centroids = {}, .bvh = {}};
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

   ctx.bvh.triangles.reserve(triangle_count);
   for (const std::uint32_t triangle : ctx.bvh.triangle_indices) {
      ctx.bvh.triangles.push_back({
         .v0 = vertex(scene, triangle, 0),
         .v1 = vertex(scene, triangle, 1),
         .v2 = vertex(scene, triangle, 2),
      });
   }
   return ctx.bvh;
}

BvhStats compute_stats(const Bvh& bvh) {
   BvhStats stats{.node_count = bvh.nodes.size()};
   if (bvh.nodes.empty()) {
      return stats;
   }
   // Parcours en profondeur avec une pile explicite : (nœud, profondeur)
   std::size_t depth_sum = 0;
   std::size_t triangle_sum = 0;
   std::vector<std::pair<std::uint32_t, std::size_t>> stack{{0, 0}};
   while (!stack.empty()) {
      const auto [index, depth] = stack.back();
      stack.pop_back();
      const BvhNode& node = bvh.nodes[index];
      if (node.count > 0) {
         ++stats.leaf_count;
         depth_sum += depth;
         triangle_sum += node.count;
         stats.max_depth = std::max(stats.max_depth, depth);
         continue;
      }
      stack.emplace_back(node.first, depth + 1);
      stack.emplace_back(node.first + 1, depth + 1);
   }
   stats.average_depth = static_cast<double>(depth_sum) / static_cast<double>(stats.leaf_count);
   stats.average_leaf_size = static_cast<double>(triangle_sum) /
                             static_cast<double>(stats.leaf_count);
   return stats;
}

} // namespace rt
