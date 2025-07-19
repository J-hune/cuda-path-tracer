#include "gpu/cuda_check.h"
#include "gpu/gl_pixel_buffer.h"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <GL/gl.h>
#include <cstddef>
#include <iostream>

GlPixelBuffer::GlPixelBuffer(unsigned int gl_buffer) {
   RT_CUDA_CHECK(
      cudaGraphicsGLRegisterBuffer(&resource_, gl_buffer, cudaGraphicsRegisterFlagsWriteDiscard)
   );
}

GlPixelBuffer::~GlPixelBuffer() {
   const cudaError_t status = cudaGraphicsUnregisterResource(resource_);
   if (status != cudaSuccess && status != cudaErrorCudartUnloading) {
      std::cerr << "CUDA error during cudaGraphicsUnregisterResource: " << cudaGetErrorName(status)
                << '\n';
   }
}

GlPixelBuffer::Mapping::Mapping(cudaGraphicsResource* resource):
   resource_(resource) {
   RT_CUDA_CHECK(cudaGraphicsMapResources(1, &resource_));
   std::size_t size = 0;
   RT_CUDA_CHECK(
      cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&pixels_), &size, resource_)
   );
}

GlPixelBuffer::Mapping::~Mapping() {
   const cudaError_t status = cudaGraphicsUnmapResources(1, &resource_);
   if (status != cudaSuccess && status != cudaErrorCudartUnloading) {
      std::cerr << "CUDA error during cudaGraphicsUnmapResources: " << cudaGetErrorName(status)
                << '\n';
   }
}
