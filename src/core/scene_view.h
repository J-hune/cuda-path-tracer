#pragma once

#include "core/aabb.h"
#include "core/bvh.h"
#include "core/environment.h"
#include "core/host_device.h"
#include "core/light.h"
#include "core/material.h"
#include "core/photon_map.h"
#include "core/ray.h"
#include "core/tangent.h"
#include "core/texture.h"
#include "core/triangle.h"
#include "core/vec2.h"
#include "core/vec3.h"

#include <cstdint>

namespace rt {

struct SceneView {
   const BvhNode* nodes = nullptr;
   const std::uint32_t* triangle_indices = nullptr;
   const BvhTriangle* bvh_triangles = nullptr;
   const Vec3* positions = nullptr;
   const std::uint32_t* indices = nullptr;
   const Material* materials = nullptr;
   const std::uint32_t* material_indices = nullptr;
   const std::uint32_t* emissive_triangles = nullptr;
   std::uint32_t emissive_triangle_count = 0;
   const PunctualLight* punctual_lights = nullptr;
   std::uint32_t punctual_light_count = 0;
   // Choix des lumières selon leur puissance : fonction de répartition (nombre de lumières + 1
   // entrées, de 0 à 1) et index de lumière de chaque triangle (-1 s'il n'est pas tiré).
   // nullptr : choix uniforme parmi toutes les lumières.
   const float* light_cdf = nullptr;
   const std::int32_t* triangle_light = nullptr;
   const Vec2* uvs = nullptr;
   const Vec3* normals = nullptr;     // par sommet, nullptr : normales géométriques
   const Tangent* tangents = nullptr; // par sommet, nullptr : pas de normal mapping
   const Texture* textures = nullptr;
   const Vec3* texels = nullptr;
   Environment environment;
   // HDRI, prioritaire sur le dégradé quand elle est présente : la dernière des lumières
   EnvironmentMap environment_map;
   // Combinaison de l'éclairage direct et du tirage de la BSDF par MIS (sinon : éclairage direct
   // seul)
   bool use_mis = true;
   // Écrêtage des fireflies, 0 : désactivé
   float firefly_clamp = 0.0f;
   // Photons de caustique, vide : caustiques laissées au path tracer seul
   PhotonMapView caustics;
   // Sphères vers lesquelles les photons sont émis, aucune : dans toutes les directions
   const PhotonTarget* photon_targets = nullptr;
   std::uint32_t photon_target_count = 0;
};

struct Hit {
   float t;
   std::uint32_t triangle;
   float u;
   float v;
};

// Parcours du BVH : le plus proche des deux enfants est visité d'abord, l'autre est empilé. Avec
// any_hit, le parcours s'arrête au premier triangle touché avant t_max (rayons d'ombre).
template <bool any_hit>
RT_HOST_DEVICE constexpr bool traverse(Ray ray, SceneView scene, float t_max, Hit& hit) {
   hit.t = t_max;
   // nodes vaut nullptr si la scène est vide
   if (scene.nodes == nullptr) {
      return false;
   }
   const Vec3 inv_dir{1.f / ray.direction.x, 1.f / ray.direction.y, 1.f / ray.direction.z};
   if (aabb_entry(ray, inv_dir, scene.nodes[0].bounds, hit.t) == kInfinity) {
      return false;
   }

   std::uint32_t stack[64];
   std::uint32_t stack_size = 0;
   std::uint32_t node_index = 0;
   bool found = false;
   while (true) {
      const BvhNode& node = scene.nodes[node_index];
      if (node.count > 0) {
         for (std::uint32_t i = node.first; i < node.first + node.count; ++i) {
            const BvhTriangle triangle = scene.bvh_triangles[i];
            TriangleHit triangle_hit{};
            if (intersect_triangle(ray, triangle.v0, triangle.v1, triangle.v2, triangle_hit) &&
                triangle_hit.t < hit.t) {
               hit = {triangle_hit.t, scene.triangle_indices[i], triangle_hit.u, triangle_hit.v};
               found = true;
               if constexpr (any_hit) {
                  return true;
               }
            }
         }
      } else {
         const std::uint32_t left = node.first;
         const std::uint32_t right = node.first + 1;
         const float t_left = aabb_entry(ray, inv_dir, scene.nodes[left].bounds, hit.t);
         const float t_right = aabb_entry(ray, inv_dir, scene.nodes[right].bounds, hit.t);
         if (t_left != kInfinity && t_right != kInfinity) {
            const bool left_first = t_left <= t_right;
            stack[stack_size++] = left_first ? right : left;
            node_index = left_first ? left : right;
            continue;
         }
         if (t_left != kInfinity || t_right != kInfinity) {
            node_index = t_left != kInfinity ? left : right;
            continue;
         }
      }

      // Nœud suivant sur la pile, en sautant ceux qui sont plus loin que le triangle trouvé
      bool next = false;
      while (stack_size > 0 && !next) {
         node_index = stack[--stack_size];
         next = aabb_entry(ray, inv_dir, scene.nodes[node_index].bounds, hit.t) != kInfinity;
      }
      if (!next) {
         return found;
      }
   }
}

// Intersection la plus proche
RT_HOST_DEVICE constexpr bool intersect_scene(Ray ray, SceneView scene, Hit& hit) {
   return traverse<false>(ray, scene, kInfinity, hit);
}

// Vrai si un triangle coupe le rayon avant t_max
RT_HOST_DEVICE constexpr bool intersects_before(Ray ray, SceneView scene, float t_max) {
   Hit hit{};
   return traverse<true>(ray, scene, t_max, hit);
}

} // namespace rt
