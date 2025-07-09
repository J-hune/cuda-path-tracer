#include "core/camera.h"
#include "core/ray.h"
#include "core/sky.h"
#include "core/triangle.h"
#include "core/vec3.h"
#include "gpu/cuda_check.h"
#include "gpu/device_buffer.h"
#include "gpu/render.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

__global__ void render_kernel(rt::Camera camera, rt::Vec3* image, int width, int height) {
   const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
   const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

   if (x >= width || y >= height) {
      return;
   }

   // On déclare un triangle devant la caméra pour tester le rendu
   const rt::Vec3 v0{-1.0f, -1.0f, -2.0f};
   const rt::Vec3 v1{1.0f, -1.0f, -2.0f};
   const rt::Vec3 v2{0.0f, 1.0f, -2.0f};
   rt::TriangleHit hit{};

   const rt::Ray ray = camera.generate_ray(
      static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f
   ); // Le + 0.5 permet de viser le centre du pixel

   if (rt::intersect_triangle(ray, v0, v1, v2, hit)) {
      image[y * width + x] = rt::Vec3{1.0f, 0.0f, 0.0f}; // Rouge si intersection
      return;
   }

   image[y * width + x] = rt::sky_color(ray.direction);
}

std::vector<rt::Vec3> render(const rt::Camera& camera, int width, int height) {
   DeviceBuffer<rt::Vec3> image(static_cast<std::size_t>(width) * height);

   const dim3 block(16, 16);
   const dim3 grid((width + 15) / 16, (height + 15) / 16);
   render_kernel<<<grid, block>>>(camera, image.data(), width, height);
   RT_CUDA_CHECK_LAUNCH();

   std::vector<rt::Vec3> host_image(static_cast<std::size_t>(width) * height);
   image.download(host_image);

   return host_image;
}
