#include "scene/tangents.h"

#include "core/tangent.h"
#include "core/vec2.h"
#include "core/vec3.h"
#include "scene/gltf_loader.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rt {

void compute_tangents(
   Scene& scene,
   std::uint32_t first_vertex,
   std::uint32_t vertex_count,
   std::size_t index_count
) {
   std::vector<Vec3> along_u(vertex_count, Vec3{0.0f, 0.0f, 0.0f});
   std::vector<Vec3> along_v(vertex_count, Vec3{0.0f, 0.0f, 0.0f});
   const std::size_t first_index = scene.indices.size() - index_count;
   for (std::size_t i = first_index; i < scene.indices.size(); i += 3) {
      const std::uint32_t a = scene.indices[i];
      const std::uint32_t b = scene.indices[i + 1];
      const std::uint32_t c = scene.indices[i + 2];
      const Vec3 e1 = scene.positions[b] - scene.positions[a];
      const Vec3 e2 = scene.positions[c] - scene.positions[a];
      const Vec2 d1 = scene.uvs[b] - scene.uvs[a];
      const Vec2 d2 = scene.uvs[c] - scene.uvs[a];
      const float det = (d1.x * d2.y) - (d2.x * d1.y);
      if (std::fabs(det) < 1e-12f) {
         continue;
      }
      const Vec3 du = ((e1 * d2.y) - (e2 * d1.y)) / det;
      const Vec3 dv = ((e2 * d1.x) - (e1 * d2.x)) / det;
      for (const std::uint32_t v : {a, b, c}) {
         along_u[v - first_vertex] += du;
         along_v[v - first_vertex] += dv;
      }
   }
   for (std::uint32_t k = 0; k < vertex_count; ++k) {
      const Vec3 n = scene.normals[first_vertex + k];
      Tangent tangent;
      if (length_squared(n) > 0.0f) {
         const Vec3 t = along_u[k] - (n * dot(n, along_u[k]));
         if (length_squared(t) > 1e-20f) {
            tangent.direction = normalize(t);
            tangent.sign = dot(cross(n, tangent.direction), -along_v[k]) < 0.0f ? -1.0f : 1.0f;
         }
      }
      scene.tangents.push_back(tangent);
   }
}

} // namespace rt
