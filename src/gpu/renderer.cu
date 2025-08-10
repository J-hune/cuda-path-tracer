#include "core/bvh.h"
#include "core/camera.h"
#include "core/environment.h"
#include "core/light.h"
#include "core/material.h"
#include "core/path_tracer.h"
#include "core/photon_map.h"
#include "core/ray.h"
#include "core/rng.h"
#include "core/scene_view.h"
#include "core/texture.h"
#include "core/tonemap.h"
#include "core/vec2.h"
#include "core/vec3.h"
#include "gpu/cuda_check.h"
#include "gpu/denoiser.h"
#include "gpu/device_buffer.h"
#include "gpu/gl_pixel_buffer.h"
#include "gpu/gpu_timer.h"
#include "gpu/photon_map.h"
#include "gpu/renderer.h"
#include "scene/bvh_builder.h"
#include "scene/gltf_loader.h"
#include "scene/photon_targets.h"

#include <cuda_runtime.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>
#include <vector_functions.h>

namespace {

// Ajoute un échantillon par pixel (l'échantillon numéro sample_index) au tampon d'accumulation
constexpr std::uint32_t kPhotonsPerPass = 1u << 17;
constexpr float kInitialRadiusFraction = 0.002f; // de la diagonale de la scène
constexpr float kInitialRadiusPixels = 1.5f;     // caustiques vues depuis la caméra
constexpr float kRadiusAlpha = 2.0f / 3.0f;

// Tampons d'accumulation : somme des échantillons de chaque pixel, pour chaque couche
struct Accumulation {
   rt::Vec3* color;
   rt::Vec3* albedo;
   rt::Vec3* normal; // dans le repère de la caméra
};

// Ajoute un échantillon par pixel (l'échantillon numéro sample_index) aux tampons d'accumulation
__global__ void accumulate_kernel(
   rt::Camera camera,
   rt::SceneView view,
   Accumulation accumulation,
   std::uint32_t sample_index,
   unsigned long long* ray_counter,
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
   std::uint32_t ray_count = 0;
   rt::FirstHit first_hit;
   const rt::Vec3 radiance = rt::trace_path(ray, view, rng, ray_count, &first_hit);
   const rt::Vec3 normal = camera.to_camera_space(first_hit.normal);

   const auto add = [&](rt::Vec3* buffer, rt::Vec3 value) {
      buffer[pixel_index] = sample_index == 0 ? value : buffer[pixel_index] + value;
   };
   add(accumulation.color, radiance);
   add(accumulation.albedo, first_hit.albedo);
   add(accumulation.normal, normal);
   atomicAdd(ray_counter, static_cast<unsigned long long>(ray_count));
}

// Moyenne des trois couches accumulées, entrée du débruiteur
__global__ void average_kernel(
   Accumulation sums,
   Accumulation means,
   float inverse_sample_count,
   int pixel_count
) {
   const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
   if (i >= pixel_count) {
      return;
   }
   means.color[i] = sums.color[i] * inverse_sample_count;
   means.albedo[i] = sums.albedo[i] * inverse_sample_count;
   means.normal[i] = sums.normal[i] * inverse_sample_count;
}

// Débruitage aux échantillons 1, 2, 4, …, 64, puis tous les 64 : l'image se nettoie vite au
// début, et le coût reste faible ensuite
bool is_denoising_step(std::uint32_t sample_count) {
   return std::has_single_bit(sample_count) || sample_count % 64 == 0;
}

__device__ unsigned char to_byte(float c) {
   return static_cast<unsigned char>(std::lround(std::fmin(std::fmax(c, 0.0f), 1.0f) * 255.0f));
}

// Moyenne d'une couche accumulée -> RGBA8 dans le pixel buffer affiché.
// Couleur : ACES puis sRGB. Albédo : sRGB. Normale : [-1, 1] ramené dans [0, 1].
__global__ void encode_kernel(
   const rt::Vec3* layer,
   Renderer::DisplayLayer kind,
   rt::DisplaySettings display,
   float inverse_sample_count,
   uchar4* pixels,
   int pixel_count
) {
   const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
   if (i >= pixel_count) {
      return;
   }
   const rt::Vec3 value = layer[i] * inverse_sample_count;
   rt::Vec3 c{0.0f, 0.0f, 0.0f};
   switch (kind) {
      case Renderer::DisplayLayer::Color:
         c = rt::display_color(value, display);
         break;
      case Renderer::DisplayLayer::Albedo:
         c = {
            rt::linear_to_srgb(std::fmin(value.x, 1.0f)),
            rt::linear_to_srgb(std::fmin(value.y, 1.0f)),
            rt::linear_to_srgb(std::fmin(value.z, 1.0f))
         };
         break;
      case Renderer::DisplayLayer::Normal:
         c = (value * 0.5f) + rt::Vec3{0.5f, 0.5f, 0.5f};
         break;
   }
   pixels[i] = make_uchar4(to_byte(c.x), to_byte(c.y), to_byte(c.z), 255);
}

} // namespace

struct Renderer::Impl {
   Impl(const rt::Scene& scene, const rt::Bvh& bvh, int width, int height):
      width(width),
      height(height),
      nodes(bvh.nodes.size()),
      triangle_indices(bvh.triangle_indices.size()),
      bvh_triangles(bvh.triangles.size()),
      positions(scene.positions.size()),
      indices(scene.indices.size()),
      materials(scene.materials.size()),
      material_indices(scene.material_indices.size()),
      emissive_triangles(scene.emissive_triangles.size()),
      punctual_lights(scene.punctual_lights.size()),
      uvs(scene.uvs.size()),
      normals(scene.normals.size()),
      textures(scene.textures.size()),
      texels(scene.texels.size()),
      accumulation(static_cast<std::size_t>(width) * height),
      albedo(static_cast<std::size_t>(width) * height),
      normal(static_cast<std::size_t>(width) * height),
      photon_targets(rt::photon_targets(scene)),
      environment(scene.environment),
      initial_photon_radius(
         bvh.nodes.empty()
            ? 0.0f
            : kInitialRadiusFraction * rt::length(bvh.nodes[0].bounds.max - bvh.nodes[0].bounds.min)
      ) {
      nodes.upload(bvh.nodes);
      triangle_indices.upload(bvh.triangle_indices);
      bvh_triangles.upload(bvh.triangles);
      positions.upload(scene.positions);
      indices.upload(scene.indices);
      materials.upload(scene.materials);
      material_indices.upload(scene.material_indices);
      emissive_triangles.upload(scene.emissive_triangles);
      punctual_lights.upload(scene.punctual_lights);
      uvs.upload(scene.uvs);
      normals.upload(scene.normals);
      textures.upload(scene.textures);
      texels.upload(scene.texels);
      photon_target_buffer = DeviceBuffer<rt::PhotonTarget>(photon_targets.size());
      photon_target_buffer.upload(photon_targets);
   }

   [[nodiscard]] rt::SceneView view() const {
      return {
         .nodes = nodes.data(),
         .triangle_indices = triangle_indices.data(),
         .bvh_triangles = bvh_triangles.data(),
         .positions = positions.data(),
         .indices = indices.data(),
         .materials = materials.data(),
         .material_indices = material_indices.data(),
         .emissive_triangles = emissive_triangles.data(),
         .emissive_triangle_count = static_cast<std::uint32_t>(emissive_triangles.size()),
         .punctual_lights = punctual_lights.data(),
         .punctual_light_count = static_cast<std::uint32_t>(punctual_lights.size()),
         .uvs = uvs.data(),
         .normals = normals.size() == 0 ? nullptr : normals.data(),
         .textures = textures.data(),
         .texels = texels.data(),
         .environment = environment,
         .use_mis = use_mis,
         .firefly_clamp = firefly_clamp,
         .caustics = {},
         .photon_targets = photon_target_buffer.data(),
         .photon_target_count = static_cast<std::uint32_t>(photon_targets.size()),
      };
   }

   int width;
   int height;
   DeviceBuffer<rt::BvhNode> nodes;
   DeviceBuffer<std::uint32_t> triangle_indices;
   DeviceBuffer<rt::BvhTriangle> bvh_triangles;
   DeviceBuffer<rt::Vec3> positions;
   DeviceBuffer<std::uint32_t> indices;
   DeviceBuffer<rt::Material> materials;
   DeviceBuffer<std::uint32_t> material_indices;
   DeviceBuffer<std::uint32_t> emissive_triangles;
   DeviceBuffer<rt::PunctualLight> punctual_lights;
   DeviceBuffer<rt::Vec2> uvs;
   DeviceBuffer<rt::Vec3> normals;
   DeviceBuffer<rt::Texture> textures;
   DeviceBuffer<rt::Vec3> texels;
   DeviceBuffer<rt::Vec3> accumulation;
   DeviceBuffer<rt::Vec3> albedo;
   DeviceBuffer<rt::Vec3> normal;
   DisplayLayer display_layer = DisplayLayer::Color;
   rt::DisplaySettings display;
   std::uint32_t sample_count = 0;
   DeviceBuffer<unsigned long long> ray_counter{1};
   GpuTimer timer;
   FrameStats last_frame;
   AccumulatedStats accumulated;
   std::vector<rt::PhotonTarget> photon_targets;
   DeviceBuffer<rt::PhotonTarget> photon_target_buffer{0};
   rt::Environment environment;
   bool use_mis = true;
   float firefly_clamp = 0.0f;
   std::optional<GlPixelBuffer> pixel_buffer;

   // Débruiteur et ses images, créés à la première utilisation
   struct Denoising {
      Denoising(int width, int height):
         denoiser(width, height),
         color(static_cast<std::size_t>(width) * height),
         albedo(static_cast<std::size_t>(width) * height),
         normal(static_cast<std::size_t>(width) * height),
         output(static_cast<std::size_t>(width) * height) {}

      Denoiser denoiser;
      DeviceBuffer<rt::Vec3> color;
      DeviceBuffer<rt::Vec3> albedo;
      DeviceBuffer<rt::Vec3> normal;
      DeviceBuffer<rt::Vec3> output;
   };
   // Caustiques par photons, carte reconstruite à chaque échantillon avec un rayon qui diminue
   // (Knaus et Zwicker, « Progressive Photon Mapping: A Probabilistic Approach », 2011)
   bool caustics = true;
   float initial_photon_radius;
   float photon_radius2 = 0.0f;
   std::optional<PhotonMap> photon_map;

   bool denoise = false;
   std::optional<Denoising> denoising;
   std::uint32_t denoised_sample_count = 0; // 0 : pas d'image débruitée depuis le reset

   [[nodiscard]] Accumulation sums() {
      return {.color = accumulation.data(), .albedo = albedo.data(), .normal = normal.data()};
   }

   // Débruite la moyenne accumulée actuelle dans denoising->output
   void run_denoiser() {
      if (!denoising) {
         denoising.emplace(width, height);
      }
      const int pixel_count = width * height;
      average_kernel<<<(pixel_count + 255) / 256, 256>>>(
         sums(),
         Accumulation{
            .color = denoising->color.data(),
            .albedo = denoising->albedo.data(),
            .normal = denoising->normal.data(),
         },
         1.0f / static_cast<float>(sample_count),
         pixel_count
      );
      RT_CUDA_CHECK_LAUNCH();
      denoising->denoiser.denoise(
         denoising->color.data(),
         denoising->albedo.data(),
         denoising->normal.data(),
         denoising->output.data()
      );
      denoised_sample_count = sample_count;
   }
};

Renderer::Renderer(const rt::Scene& scene, const rt::Bvh& bvh, int width, int height):
   impl_(std::make_unique<Impl>(scene, bvh, width, height)) {}

Renderer::~Renderer() = default;
Renderer::Renderer(Renderer&&) noexcept = default;
Renderer& Renderer::operator=(Renderer&&) noexcept = default;

void Renderer::attach_pixel_buffer(unsigned int gl_buffer) {
   impl_->pixel_buffer.emplace(gl_buffer);
}

void Renderer::detach_pixel_buffer() {
   impl_->pixel_buffer.reset();
}

void Renderer::resize(int width, int height) {
   impl_->width = width;
   impl_->height = height;
   impl_->accumulation = DeviceBuffer<rt::Vec3>(static_cast<std::size_t>(width) * height);
   impl_->albedo = DeviceBuffer<rt::Vec3>(static_cast<std::size_t>(width) * height);
   impl_->normal = DeviceBuffer<rt::Vec3>(static_cast<std::size_t>(width) * height);
   impl_->denoising.reset();
   reset();
}

void Renderer::set_mis(bool enabled) {
   impl_->use_mis = enabled;
   reset();
}

bool Renderer::mis() const {
   return impl_->use_mis;
}

void Renderer::set_firefly_clamp(float max_contribution) {
   impl_->firefly_clamp = max_contribution;
   reset();
}

float Renderer::firefly_clamp() const {
   return impl_->firefly_clamp;
}

void Renderer::set_display(rt::DisplaySettings display) {
   impl_->display = display;
}

rt::DisplaySettings Renderer::display() const {
   return impl_->display;
}

void Renderer::set_display_layer(DisplayLayer layer) {
   impl_->display_layer = layer;
   impl_->denoised_sample_count = 0;
}

Renderer::DisplayLayer Renderer::display_layer() const {
   return impl_->display_layer;
}

void Renderer::reset() {
   impl_->sample_count = 0;
   impl_->denoised_sample_count = 0;
}

void Renderer::set_caustics(bool enabled) {
   impl_->caustics = enabled;
   reset();
}

bool Renderer::caustics() const {
   return impl_->caustics;
}

Renderer::CausticSettings Renderer::caustic_settings() const {
   return {
      .photons_per_pass = kPhotonsPerPass,
      .initial_radius = impl_->initial_photon_radius,
      .initial_radius_pixels = kInitialRadiusPixels,
      .alpha = kRadiusAlpha,
   };
}

void Renderer::set_denoiser(bool enabled) {
   impl_->denoise = enabled;
   impl_->denoised_sample_count = 0;
}

bool Renderer::denoiser() const {
   return impl_->denoise;
}

std::vector<rt::Vec3> Renderer::denoised_image() {
   if (impl_->sample_count == 0) {
      return {};
   }
   if (impl_->denoised_sample_count != impl_->sample_count) {
      impl_->run_denoiser();
   }
   std::vector<rt::Vec3> pixels(impl_->denoising->output.size());
   impl_->denoising->output.download(pixels);
   return pixels;
}

std::uint32_t Renderer::sample_count() const {
   return impl_->sample_count;
}

std::vector<rt::Vec3> Renderer::image(DisplayLayer layer) const {
   const DeviceBuffer<rt::Vec3>& source = layer == DisplayLayer::Albedo   ? impl_->albedo
                                          : layer == DisplayLayer::Normal ? impl_->normal
                                                                          : impl_->accumulation;
   std::vector<rt::Vec3> pixels(source.size());
   source.download(pixels);
   const float inverse_count = 1.0f / static_cast<float>(impl_->sample_count);
   for (rt::Vec3& pixel : pixels) {
      pixel *= inverse_count;
   }
   return pixels;
}

Renderer::FrameStats Renderer::last_frame() const {
   return impl_->last_frame;
}

Renderer::AccumulatedStats Renderer::accumulated() const {
   return impl_->accumulated;
}

Renderer::DeviceInfo Renderer::device_info() {
   int device = 0;
   RT_CUDA_CHECK(cudaGetDevice(&device));
   cudaDeviceProp properties{};
   RT_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
   DeviceInfo info{
      .name = properties.name,
      .compute_capability_major = properties.major,
      .compute_capability_minor = properties.minor,
      .memory_bytes = properties.totalGlobalMem,
   };
   RT_CUDA_CHECK(cudaDriverGetVersion(&info.driver_version));
   RT_CUDA_CHECK(cudaRuntimeGetVersion(&info.runtime_version));
   return info;
}

void Renderer::render(const rt::Camera& camera) {
   Impl& impl = *impl_;
   const dim3 block(16, 16);
   const dim3 grid((impl.width + 15) / 16, (impl.height + 15) / 16);
   RT_CUDA_CHECK(cudaMemset(impl.ray_counter.data(), 0, sizeof(unsigned long long)));
   rt::SceneView view = impl.view();
   if (impl.caustics) {
      if (impl.sample_count == 0) {
         impl.photon_radius2 = impl.initial_photon_radius * impl.initial_photon_radius;
      } else {
         const auto pass = static_cast<float>(impl.sample_count);
         impl.photon_radius2 *= (pass + kRadiusAlpha) / (pass + 1.0f);
      }
      if (!impl.photon_map) {
         impl.photon_map.emplace(kPhotonsPerPass);
      }
      impl.photon_map->build(view, impl.sample_count, std::sqrt(impl.photon_radius2));
      view.caustics = impl.photon_map->view();
      const float shrink = std::sqrt(impl.photon_radius2) / impl.initial_photon_radius;
      view.caustics.radius_per_distance = kInitialRadiusPixels * camera.pixel_angle() * shrink;
   }

   impl.timer.start();
   accumulate_kernel<<<grid, block>>>(
      camera,
      view,
      impl.sums(),
      impl.sample_count,
      impl.ray_counter.data(),
      impl.width,
      impl.height
   );
   RT_CUDA_CHECK_LAUNCH();
   impl.timer.stop();
   ++impl.sample_count;

   std::vector<unsigned long long> rays(1);
   impl.ray_counter.download(rays);
   impl.last_frame = {.milliseconds = impl.timer.elapsed_milliseconds(), .rays = rays[0]};
   if (impl.sample_count == 1) {
      impl.accumulated = {};
   }
   impl.accumulated.milliseconds += impl.last_frame.milliseconds;
   impl.accumulated.rays += impl.last_frame.rays;

   const bool show_denoised = impl.denoise && impl.display_layer == DisplayLayer::Color;
   if (show_denoised && (impl.denoised_sample_count == 0 || is_denoising_step(impl.sample_count))) {
      impl.run_denoiser();
   }

   if (impl.pixel_buffer) {
      const int pixel_count = impl.width * impl.height;
      const GlPixelBuffer::Mapping mapping = impl.pixel_buffer->map();
      const bool denoised = show_denoised && impl.denoised_sample_count > 0;
      const rt::Vec3* layer = denoised ? impl.denoising->output.data()
                              : impl.display_layer == DisplayLayer::Albedo ? impl.albedo.data()
                              : impl.display_layer == DisplayLayer::Normal
                                 ? impl.normal.data()
                                 : impl.accumulation.data();
      encode_kernel<<<(pixel_count + 255) / 256, 256>>>(
         layer,
         impl.display_layer,
         impl.display,
         denoised ? 1.0f : 1.0f / static_cast<float>(impl.sample_count),
         mapping.pixels(),
         pixel_count
      );
      RT_CUDA_CHECK_LAUNCH();
   }
}
