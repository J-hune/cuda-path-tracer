#include "core/bvh.h"
#include "core/camera.h"
#include "core/material.h"
#include "core/path_tracer.h"
#include "core/ray.h"
#include "core/rng.h"
#include "core/scene_view.h"
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
      color += rt::trace_path(ray, view, rng);
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
      .nodes = bvh_nodes.data(),
      .triangle_indices = bvh_triangle_indices.data(),
      .positions = scene_positions.data(),
      .indices = scene_indices.data(),
      .materials = scene_materials.data(),
      .material_indices = scene_material_indices.data(),
      .emissive_triangles = scene_emissive_triangles.data(),
      .emissive_triangle_count = static_cast<std::uint32_t>(scene.emissive_triangles.size()),
      .environment = scene.environment,
   };

   render_kernel<<<grid, block>>>(camera, view, image.data(), width, height);
   RT_CUDA_CHECK_LAUNCH();

   std::vector<rt::Vec3> host_image(static_cast<std::size_t>(width) * height);
   image.download(host_image);

   return host_image;
}
