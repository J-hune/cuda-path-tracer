#include "core/bvh.h"
#include "core/camera.h"
#include "core/ray.h"
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

__global__ void
render_kernel(rt::Camera camera, rt::SceneView view, rt::Vec3* image, int width, int height) {
   const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
   const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

   if (x >= width || y >= height) {
      return;
   }

   const rt::Ray ray = camera.generate_ray(
      static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f
   ); // Le + 0.5 permet de viser le centre du pixel

   rt::Hit hit{};
   if (intersect_scene(ray, view, hit)) {
      // On récupère le triangle et on calcule sa normale
      const rt::Vec3 v0 = view.positions[view.indices[hit.triangle * 3 + 0]];
      const rt::Vec3 v1 = view.positions[view.indices[hit.triangle * 3 + 1]];
      const rt::Vec3 v2 = view.positions[view.indices[hit.triangle * 3 + 2]];
      const rt::Vec3 n = rt::normalize(rt::cross(v1 - v0, v2 - v0));

      // On calcule la couleur en fonction de la normale
      image[y * width + x] = 0.5f * (n + rt::Vec3{1.0f, 1.0f, 1.0f});
      return;
   }

   image[y * width + x] = rt::sky_color(ray.direction);
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
      bvh_nodes.data(), bvh_triangle_indices.data(), scene_positions.data(), scene_indices.data()
   };

   render_kernel<<<grid, block>>>(camera, view, image.data(), width, height);
   RT_CUDA_CHECK_LAUNCH();

   std::vector<rt::Vec3> host_image(static_cast<std::size_t>(width) * height);
   image.download(host_image);

   return host_image;
}
