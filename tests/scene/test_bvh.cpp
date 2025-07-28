#include "core/aabb.h"
#include "core/bvh.h"
#include "core/vec3.h"
#include "scene/bvh_builder.h"
#include "scene/gltf_loader.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

// Grille de n × n petits triangles dans le plan z = 0
rt::Scene make_grid(std::uint32_t n) {
   rt::Scene scene;
   for (std::uint32_t y = 0; y < n; ++y) {
      for (std::uint32_t x = 0; x < n; ++x) {
         const auto base = static_cast<std::uint32_t>(scene.positions.size());
         const auto fx = static_cast<float>(x);
         const auto fy = static_cast<float>(y);
         scene.positions.push_back({fx, fy, 0});
         scene.positions.push_back({fx + 0.5f, fy, 0});
         scene.positions.push_back({fx, fy + 0.5f, 0});
         scene.indices.insert(scene.indices.end(), {base, base + 1, base + 2});
      }
   }
   return scene;
}

bool contains(const rt::Aabb& outer, rt::Vec3 p) {
   return p.x >= outer.min.x && p.y >= outer.min.y && p.z >= outer.min.z && p.x <= outer.max.x &&
          p.y <= outer.max.y && p.z <= outer.max.z;
}

} // namespace

TEST_CASE("BVH d'une scène vide", "[bvh]") {
   const rt::Bvh bvh = rt::build_bvh(rt::Scene{});
   CHECK(bvh.nodes.empty());
   CHECK(bvh.triangle_indices.empty());
}

TEST_CASE("BVH : chaque triangle est dans exactement une feuille", "[bvh]") {
   const rt::BvhSplit split = GENERATE(rt::BvhSplit::Median, rt::BvhSplit::Sah);
   const rt::Scene scene = make_grid(16);
   const rt::Bvh bvh = rt::build_bvh(scene, split);
   const std::size_t triangle_count = scene.indices.size() / 3;

   REQUIRE(bvh.triangle_indices.size() == triangle_count);
   std::vector<int> seen(triangle_count, 0);
   for (const rt::BvhNode& node : bvh.nodes) {
      for (std::uint32_t i = node.first; i < node.first + node.count; ++i) {
         ++seen[bvh.triangle_indices[i]];
      }
   }
   for (const int s : seen) {
      REQUIRE(s == 1);
   }
}

TEST_CASE("BVH : les boîtes englobent leur contenu", "[bvh]") {
   const rt::BvhSplit split = GENERATE(rt::BvhSplit::Median, rt::BvhSplit::Sah);
   const rt::Scene scene = make_grid(16);
   const rt::Bvh bvh = rt::build_bvh(scene, split);

   for (const rt::BvhNode& node : bvh.nodes) {
      if (node.count == 0) {
         for (const std::uint32_t child : {node.first, node.first + 1}) {
            REQUIRE(contains(node.bounds, bvh.nodes[child].bounds.min));
            REQUIRE(contains(node.bounds, bvh.nodes[child].bounds.max));
         }
         continue;
      }
      for (std::uint32_t i = node.first; i < node.first + node.count; ++i) {
         const std::uint32_t triangle = bvh.triangle_indices[i];
         for (std::uint32_t corner = 0; corner < 3; ++corner) {
            REQUIRE(contains(node.bounds, scene.positions[scene.indices[(3 * triangle) + corner]]));
         }
      }
   }
}

TEST_CASE("BVH : les feuilles sont petites", "[bvh]") {
   // Médian : 2 triangles au plus. SAH : la feuille est choisie par le coût, 8 triangles au plus
   for (const auto& [split, max_leaf] :
        {std::pair{rt::BvhSplit::Median, 2u}, {rt::BvhSplit::Sah, 8u}}) {
      const rt::Bvh bvh = rt::build_bvh(make_grid(16), split);
      for (const rt::BvhNode& node : bvh.nodes) {
         REQUIRE(node.count <= max_leaf);
      }
   }
}

TEST_CASE("BVH : statistiques cohérentes", "[bvh]") {
   const rt::Scene scene = make_grid(16);
   const rt::Bvh bvh = rt::build_bvh(scene, rt::BvhSplit::Median);
   const rt::BvhStats stats = rt::compute_stats(bvh);
   // Arbre binaire complet : n feuilles pour 2n - 1 nœuds
   CHECK(stats.node_count == bvh.nodes.size());
   CHECK(stats.node_count == (2 * stats.leaf_count) - 1);
   const std::size_t triangle_count = scene.indices.size() / 3;
   CHECK(
      stats.average_leaf_size * static_cast<double>(stats.leaf_count) ==
      static_cast<double>(triangle_count)
   );
   CHECK(stats.average_depth <= static_cast<double>(stats.max_depth));
}
