#include "core/bvh.h"
#include "core/camera.h"
#include "core/environment.h"
#include "core/material.h"
#include "core/path_tracer.h"
#include "core/ray.h"
#include "core/rng.h"
#include "core/scene_view.h"
#include "core/texture.h"
#include "core/vec2.h"
#include "core/vec3.h"
#include "gpu/cuda_check.h"
#include "gpu/device_buffer.h"
#include "gpu/gl_pixel_buffer.h"
#include "gpu/renderer.h"
#include "scene/bvh_builder.h"
#include "scene/gltf_loader.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector_functions.h>

namespace {

// Ajoute un échantillon par pixel (l'échantillon numéro sample_index) au tampon d'accumulation
__global__ void accumulate_kernel(
   rt::Camera camera,
   rt::SceneView view,
   rt::Vec3* accumulation,
   std::uint32_t sample_index,
   int width,
   int height
) {
   const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
   const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
   if (x >= width || y >= height) {
      return;
   }

   const auto pixel_index = static_cast<std::uint32_t>((y * width) + x);
   rt::Pcg32 rng = rt::Pcg32::make_pixel_rng(pixel_index, sample_index);
   const float jitter_x = rng.next_float();
   const float jitter_y = rng.next_float();
   const rt::Ray ray = camera.generate_ray(
      static_cast<float>(x) + jitter_x,
      static_cast<float>(y) + jitter_y
   );
   const rt::Vec3 radiance = rt::trace_path(ray, view, rng);
   accumulation[pixel_index] = sample_index == 0 ? radiance : accumulation[pixel_index] + radiance;
}

__device__ unsigned char encode_channel(float linear) {
   const float clamped = std::fmin(std::fmax(linear, 0.0f), 1.0f);
   return static_cast<unsigned char>(std::lround(std::pow(clamped, 1.0f / 2.2f) * 255.0f));
}

// Moyenne des échantillons accumulés -> RGBA8 encodé gamma, dans le pixel buffer affiché
__global__ void encode_kernel(
   const rt::Vec3* accumulation,
   float inverse_sample_count,
   uchar4* pixels,
   int pixel_count
) {
   const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
   if (i >= pixel_count) {
      return;
   }
   const rt::Vec3 c = accumulation[i] * inverse_sample_count;
   pixels[i] = make_uchar4(encode_channel(c.x), encode_channel(c.y), encode_channel(c.z), 255);
}

} // namespace

struct Renderer::Impl {
   Impl(const rt::Scene& scene, const rt::Bvh& bvh, int width, int height):
      width(width),
      height(height),
      nodes(bvh.nodes.size()),
      triangle_indices(bvh.triangle_indices.size()),
      positions(scene.positions.size()),
      indices(scene.indices.size()),
      materials(scene.materials.size()),
      material_indices(scene.material_indices.size()),
      emissive_triangles(scene.emissive_triangles.size()),
      uvs(scene.uvs.size()),
      textures(scene.textures.size()),
      texels(scene.texels.size()),
      accumulation(static_cast<std::size_t>(width) * height),
      environment(scene.environment) {
      nodes.upload(bvh.nodes);
      triangle_indices.upload(bvh.triangle_indices);
      positions.upload(scene.positions);
      indices.upload(scene.indices);
      materials.upload(scene.materials);
      material_indices.upload(scene.material_indices);
      emissive_triangles.upload(scene.emissive_triangles);
      uvs.upload(scene.uvs);
      textures.upload(scene.textures);
      texels.upload(scene.texels);
   }

   [[nodiscard]] rt::SceneView view() const {
      return {
         .nodes = nodes.data(),
         .triangle_indices = triangle_indices.data(),
         .positions = positions.data(),
         .indices = indices.data(),
         .materials = materials.data(),
         .material_indices = material_indices.data(),
         .emissive_triangles = emissive_triangles.data(),
         .emissive_triangle_count = static_cast<std::uint32_t>(emissive_triangles.size()),
         .uvs = uvs.data(),
         .textures = textures.data(),
         .texels = texels.data(),
         .environment = environment,
      };
   }

   int width;
   int height;
   DeviceBuffer<rt::BvhNode> nodes;
   DeviceBuffer<std::uint32_t> triangle_indices;
   DeviceBuffer<rt::Vec3> positions;
   DeviceBuffer<std::uint32_t> indices;
   DeviceBuffer<rt::Material> materials;
   DeviceBuffer<std::uint32_t> material_indices;
   DeviceBuffer<std::uint32_t> emissive_triangles;
   DeviceBuffer<rt::Vec2> uvs;
   DeviceBuffer<rt::Texture> textures;
   DeviceBuffer<rt::Vec3> texels;
   DeviceBuffer<rt::Vec3> accumulation;
   std::uint32_t sample_count = 0;
   rt::Environment environment;
   std::optional<GlPixelBuffer> pixel_buffer;
};

Renderer::Renderer(const rt::Scene& scene, const rt::Bvh& bvh, int width, int height):
   impl_(std::make_unique<Impl>(scene, bvh, width, height)) {}

Renderer::~Renderer() = default;
Renderer::Renderer(Renderer&&) noexcept = default;
Renderer& Renderer::operator=(Renderer&&) noexcept = default;

void Renderer::attach_pixel_buffer(unsigned int gl_buffer) {
   impl_->pixel_buffer.emplace(gl_buffer);
}

void Renderer::reset() {
   impl_->sample_count = 0;
}

std::uint32_t Renderer::sample_count() const {
   return impl_->sample_count;
}

void Renderer::render(const rt::Camera& camera) {
   Impl& impl = *impl_;
   const dim3 block(16, 16);
   const dim3 grid((impl.width + 15) / 16, (impl.height + 15) / 16);
   accumulate_kernel<<<grid, block>>>(
      camera,
      impl.view(),
      impl.accumulation.data(),
      impl.sample_count,
      impl.width,
      impl.height
   );
   RT_CUDA_CHECK_LAUNCH();
   ++impl.sample_count;

   if (impl.pixel_buffer) {
      const int pixel_count = impl.width * impl.height;
      const GlPixelBuffer::Mapping mapping = impl.pixel_buffer->map();
      encode_kernel<<<(pixel_count + 255) / 256, 256>>>(
         impl.accumulation.data(),
         1.0f / static_cast<float>(impl.sample_count),
         mapping.pixels(),
         pixel_count
      );
      RT_CUDA_CHECK_LAUNCH();
   }
}
