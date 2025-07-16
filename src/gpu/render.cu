#include "core/bvh.h"
#include "core/camera.h"
#include "core/material.h"
#include "core/ray.h"
#include "core/rng.h"
#include "core/sampling.h"
#include "core/scene_view.h"
#include "core/sky.h"
#include "core/vec3.h"
#include "gpu/cuda_check.h"
#include "gpu/device_buffer.h"
#include "gpu/render.h"
#include "scene/bvh_builder.h"
#include "scene/gltf_loader.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

constexpr int kSamplesPerPixel = 16;
constexpr int kMaxBounces = 4;
constexpr float kRayOffset = 1e-3f;

// Normale géométrique du triangle, orientée vers le côté d'où vient le rayon
__device__ rt::Vec3 facing_normal(rt::SceneView view, std::uint32_t triangle, rt::Vec3 direction) {
   const rt::Vec3 v0 = view.positions[view.indices[(3 * triangle) + 0]];
   const rt::Vec3 v1 = view.positions[view.indices[(3 * triangle) + 1]];
   const rt::Vec3 v2 = view.positions[view.indices[(3 * triangle) + 2]];
   const rt::Vec3 n = rt::normalize(rt::cross(v1 - v0, v2 - v0));
   return rt::dot(n, direction) > 0.0f ? -n : n;
}

// Vérifie si un objet se trouve entre l'origine et la cible
__device__ bool is_occluded(rt::SceneView view, rt::Vec3 origin, rt::Vec3 target) {
   const rt::Vec3 direction = target - origin;
   const float distance = rt::length(direction);
   const rt::Ray ray{origin, rt::normalize(direction)};
   rt::Hit hit{};

   // La lampe visée ne compte pas comme obstacle
   return rt::intersect_scene(ray, view, hit) && hit.t < distance - kRayOffset;
}

// Échantillonne un triangle émissif au hasard et calcule la contribution de la lumière directe
__device__ rt::Vec3 direct_light(
   rt::SceneView view,
   rt::Vec3 x,
   rt::Vec3 n,
   rt::Vec3 albedo,
   rt::Pcg32& rng
) {
   if (view.emissive_triangle_count == 0) {
      return rt::Vec3{0.0f, 0.0f, 0.0f};
   }

   // Choix uniforme d'une lampe, index borné à count - 1
   const auto picked = static_cast<std::uint32_t>(
      rng.next_float() * static_cast<float>(view.emissive_triangle_count)
   );
   const std::uint32_t k = picked < view.emissive_triangle_count ? picked
                                                                 : view.emissive_triangle_count - 1;
   const std::uint32_t tri = view.emissive_triangles[k];
   const rt::Vec3 v0 = view.positions[view.indices[(3 * tri) + 0]];
   const rt::Vec3 v1 = view.positions[view.indices[(3 * tri) + 1]];
   const rt::Vec3 v2 = view.positions[view.indices[(3 * tri) + 2]];
   const rt::Vec3 y = rt::sample_point_on_triangle(v0, v1, v2, rng);

   const rt::Vec3 cross_l = rt::cross(v1 - v0, v2 - v0);
   const float area = 0.5f * rt::length(cross_l);
   const rt::Vec3 n_l = rt::normalize(cross_l);

   const float d2 = rt::length_squared(y - x);
   const rt::Vec3 wi = (y - x) / std::sqrt(d2);

   const float cos_x = rt::dot(n, wi);
   const float cos_y = std::fabs(rt::dot(n_l, wi));

   if (cos_x <= 0.0f || cos_y <= 0.0f || is_occluded(view, x + (n * kRayOffset), y)) {
      return rt::Vec3{0.0f, 0.0f, 0.0f};
   }

   const rt::Material material = view.materials[view.material_indices[tri]];
   // BSDF lambertienne = albedo / π ; densité du point tiré = 1 / (aire × nombre de lampes)
   const float geometry = cos_x * cos_y / d2;
   const float inverse_pdf = area * static_cast<float>(view.emissive_triangle_count);
   return (albedo / rt::kPi) * material.emission * (geometry * inverse_pdf);
}

// Suit un chemin à partir du rayon caméra et renvoie la lumière qu'il rapporte
__device__ rt::Vec3 trace_path(rt::Ray ray, rt::SceneView view, rt::Pcg32& rng) {
   rt::Vec3 radiance{0.0f, 0.0f, 0.0f};
   rt::Vec3 throughput{1.0f, 1.0f, 1.0f};

   for (int bounce = 0; bounce < kMaxBounces; ++bounce) {
      rt::Hit hit{};
      if (!rt::intersect_scene(ray, view, hit)) {
         radiance += throughput * rt::sky_color(ray.direction);
         return radiance;
      }

      const rt::Material material = view.materials[view.material_indices[hit.triangle]];
      if (bounce == 0) {
         radiance += throughput * material.emission;
      }
      const rt::Vec3 point = ray.at(hit.t);
      const rt::Vec3 n = facing_normal(view, hit.triangle, ray.direction);
      const rt::Vec3 direct = direct_light(view, point, n, material.albedo, rng);
      radiance += throughput * direct;
      throughput = throughput * material.albedo;

      ray = {point + (n * kRayOffset), rt::sample_cosine_hemisphere(n, rng)};
   }

   return radiance;
}

} // namespace

__global__ void render_kernel(
   rt::Camera camera,
   rt::SceneView view,
   rt::Vec3* image,
   int width,
   int height
) {
   const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
   const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
   if (x >= width || y >= height) {
      return;
   }

   const auto pixel_index = static_cast<std::uint32_t>((y * width) + x);
   rt::Vec3 color{0.0f, 0.0f, 0.0f};
   for (int sample = 0; sample < kSamplesPerPixel; ++sample) {
      rt::Pcg32 rng = rt::Pcg32::make_pixel_rng(pixel_index, static_cast<std::uint32_t>(sample));
      const float jitter_x = rng.next_float();
      const float jitter_y = rng.next_float();
      const rt::Ray ray = camera.generate_ray(
         static_cast<float>(x) + jitter_x,
         static_cast<float>(y) + jitter_y
      );
      color += trace_path(ray, view, rng);
   }

   image[pixel_index] = color / static_cast<float>(kSamplesPerPixel);
}

std::vector<rt::Vec3> render(
   const rt::Camera& camera,
   const rt::Scene& scene,
   const rt::Bvh& bvh,
   int width,
   int height
) {
   DeviceBuffer<rt::Vec3> image(static_cast<std::size_t>(width) * height);

   const dim3 block(16, 16);
   const dim3 grid((width + 15) / 16, (height + 15) / 16);

   DeviceBuffer<rt::BvhNode> bvh_nodes(bvh.nodes.size());
   DeviceBuffer<std::uint32_t> bvh_triangle_indices(bvh.triangle_indices.size());
   DeviceBuffer<rt::Vec3> scene_positions(scene.positions.size());
   DeviceBuffer<std::uint32_t> scene_indices(scene.indices.size());
   DeviceBuffer<rt::Material> scene_materials(scene.materials.size());
   DeviceBuffer<std::uint32_t> scene_material_indices(scene.material_indices.size());
   DeviceBuffer<std::uint32_t> scene_emissive_triangles(scene.emissive_triangles.size());

   bvh_nodes.upload(bvh.nodes);
   bvh_triangle_indices.upload(bvh.triangle_indices);
   scene_positions.upload(scene.positions);
   scene_indices.upload(scene.indices);
   scene_materials.upload(scene.materials);
   scene_material_indices.upload(scene.material_indices);
   scene_emissive_triangles.upload(scene.emissive_triangles);

   const rt::SceneView view{
      bvh_nodes.data(),
      bvh_triangle_indices.data(),
      scene_positions.data(),
      scene_indices.data(),
      scene_materials.data(),
      scene_material_indices.data(),
      scene_emissive_triangles.data(),
      static_cast<std::uint32_t>(scene.emissive_triangles.size())
   };

   render_kernel<<<grid, block>>>(camera, view, image.data(), width, height);
   RT_CUDA_CHECK_LAUNCH();

   std::vector<rt::Vec3> host_image(static_cast<std::size_t>(width) * height);
   image.download(host_image);

   return host_image;
}
