#include "core/bvh.h"
#include "core/camera.h"
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

#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

constexpr int kSamplesPerPixel = 16;
constexpr int kMaxBounces = 4;
constexpr float kAlbedo = 0.8f;
constexpr float kRayOffset = 1e-3f;

// Normale géométrique du triangle, orientée vers le côté d'où vient le rayon
__device__ rt::Vec3 facing_normal(rt::SceneView view, std::uint32_t triangle, rt::Vec3 direction) {
   const rt::Vec3 v0 = view.positions[view.indices[(3 * triangle) + 0]];
   const rt::Vec3 v1 = view.positions[view.indices[(3 * triangle) + 1]];
   const rt::Vec3 v2 = view.positions[view.indices[(3 * triangle) + 2]];
   const rt::Vec3 n = rt::normalize(rt::cross(v1 - v0, v2 - v0));
   return rt::dot(n, direction) > 0.0f ? -n : n;
}

// Suit un chemin à partir du rayon caméra et renvoie la lumière qu'il rapporte
__device__ rt::Vec3 trace_path(rt::Ray ray, rt::SceneView view, rt::Pcg32& rng) {
   rt::Vec3 throughput{1.0f, 1.0f, 1.0f};
   for (int bounce = 0; bounce < kMaxBounces; ++bounce) {
      rt::Hit hit{};
      if (!rt::intersect_scene(ray, view, hit)) {
         return throughput * rt::sky_color(ray.direction);
      }

      const rt::Vec3 n = facing_normal(view, hit.triangle, ray.direction);
      ray = {ray.at(hit.t) + (n * kRayOffset), rt::sample_cosine_hemisphere(n, rng)};
      throughput *= kAlbedo;
   }
   return {0.0f, 0.0f, 0.0f};
}

} // namespace

__global__ void
render_kernel(rt::Camera camera, rt::SceneView view, rt::Vec3* image, int width, int height) {
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
   const rt::Camera& camera, const rt::Scene& scene, const rt::Bvh& bvh, int width, int height
) {
   DeviceBuffer<rt::Vec3> image(static_cast<std::size_t>(width) * height);

   const dim3 block(16, 16);
   const dim3 grid((width + 15) / 16, (height + 15) / 16);

   DeviceBuffer<rt::BvhNode> bvh_nodes(bvh.nodes.size());
   DeviceBuffer<std::uint32_t> bvh_triangle_indices(bvh.triangle_indices.size());
   DeviceBuffer<rt::Vec3> scene_positions(scene.positions.size());
   DeviceBuffer<std::uint32_t> scene_indices(scene.indices.size());

   bvh_nodes.upload(bvh.nodes);
   bvh_triangle_indices.upload(bvh.triangle_indices);
   scene_positions.upload(scene.positions);
   scene_indices.upload(scene.indices);

   const rt::SceneView view{
      bvh_nodes.data(),
      bvh_triangle_indices.data(),
      scene_positions.data(),
      scene_indices.data()
   };

   render_kernel<<<grid, block>>>(camera, view, image.data(), width, height);
   RT_CUDA_CHECK_LAUNCH();

   std::vector<rt::Vec3> host_image(static_cast<std::size_t>(width) * height);
   image.download(host_image);

   return host_image;
}
