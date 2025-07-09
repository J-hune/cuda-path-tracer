#include "core/camera.h"
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

   const float u = static_cast<float>(x) / static_cast<float>(width);
   const float v = static_cast<float>(y) / static_cast<float>(height);

   image[y * width + x] = rt::Vec3{u, v, 0.0f};
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
