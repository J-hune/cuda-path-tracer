#pragma once

#include "core/aabb.h"
#include "core/bvh.h"
#include "core/environment.h"
#include "core/host_device.h"
#include "core/light.h"
#include "core/material.h"
#include "core/ray.h"
#include "core/texture.h"
#include "core/triangle.h"
#include "core/vec2.h"
#include "core/vec3.h"

#include <cstdint>

namespace rt {

struct SceneView {
   const BvhNode* nodes = nullptr;
   const std::uint32_t* triangle_indices = nullptr;
   const Vec3* positions = nullptr;
   const std::uint32_t* indices = nullptr;
   const Material* materials = nullptr;
   const std::uint32_t* material_indices = nullptr;
   const std::uint32_t* emissive_triangles = nullptr;
   std::uint32_t emissive_triangle_count = 0;
   const PunctualLight* punctual_lights = nullptr;
   std::uint32_t punctual_light_count = 0;
   const Vec2* uvs = nullptr;
   const Vec3* normals = nullptr; // par sommet, nullptr : normales géométriques
   const Texture* textures = nullptr;
   const Vec3* texels = nullptr;
   Environment environment;
   // Combinaison de l'éclairage direct et du tirage de la BSDF par MIS (sinon : éclairage direct
   // seul)
   bool use_mis = true;
   // Écrêtage des fireflies, 0 : désactivé
   float firefly_clamp = 0.0f;
};

struct Hit {
   float t;
   std::uint32_t triangle;
   float u;
   float v;
};

RT_HOST_DEVICE constexpr bool intersect_scene(Ray ray, SceneView scene, Hit& hit) {
   std::uint32_t stack[64];
   std::uint32_t stack_size = 0;
   stack[stack_size++] = 0; // empiler la racine
   hit.t = kInfinity;
   const Vec3 inv_dir{1.f / ray.direction.x, 1.f / ray.direction.y, 1.f / ray.direction.z};

   // nodes vaut nullptr si la scène est vide
   if (scene.nodes == nullptr) {
      return false;
   }

   while (stack_size > 0) {
      const std::uint32_t node_index = stack[--stack_size];
      const BvhNode& node = scene.nodes[node_index];

      // Si le rayon ne touche pas la boîte du nœud, on passe au suivant
      if (!intersect_aabb(ray, inv_dir, node.bounds, hit.t)) {
         continue;
      }

      // Si c'est une feuille, on teste tous ses triangles
      if (node.count > 0) {
         for (std::uint32_t i = node.first; i < node.first + node.count; ++i) {
            const std::uint32_t triangle = scene.triangle_indices[i];
            const Vec3 v0 = scene.positions[scene.indices[3 * triangle + 0]];
            const Vec3 v1 = scene.positions[scene.indices[3 * triangle + 1]];
            const Vec3 v2 = scene.positions[scene.indices[3 * triangle + 2]];

            TriangleHit triangle_hit{};
            if (intersect_triangle(ray, v0, v1, v2, triangle_hit) && triangle_hit.t < hit.t) {
               hit = {triangle_hit.t, triangle, triangle_hit.u, triangle_hit.v};
            }
         }
      } else {
         stack[stack_size++] = node.first;     // empiler l'enfant gauche
         stack[stack_size++] = node.first + 1; // empiler l'enfant droit
      }
   }

   return hit.t < kInfinity;
}

} // namespace rt
