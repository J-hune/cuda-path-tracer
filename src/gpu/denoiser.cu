#include "core/vec3.h"
#include "gpu/cuda_check.h"
#include "gpu/denoiser.h"
#include "gpu/device_buffer.h"
#include "gpu/optix_check.h"

#include <cuda_runtime.h>

#include <iostream>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_types.h>

namespace {

CUdeviceptr device_pointer(const void* pointer) {
   return reinterpret_cast<CUdeviceptr>(pointer);
}

} // namespace

Denoiser::Denoiser(int width, int height):
   width_(width),
   height_(height) {
   // Le contexte OptiX s'appuie sur le contexte CUDA courant, créé ici s'il n'existe pas encore
   RT_CUDA_CHECK(cudaFree(nullptr));
   RT_OPTIX_CHECK(optixInit());
   RT_OPTIX_CHECK(optixDeviceContextCreate(nullptr, nullptr, &context_));

   const OptixDenoiserOptions options{
      .guideAlbedo = 1,
      .guideNormal = 1,
      .denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY,
   };
   RT_OPTIX_CHECK(
      optixDenoiserCreate(context_, OPTIX_DENOISER_MODEL_KIND_AOV, &options, &denoiser_)
   );

   OptixDenoiserSizes sizes{};
   RT_OPTIX_CHECK(optixDenoiserComputeMemoryResources(
      denoiser_,
      static_cast<unsigned int>(width),
      static_cast<unsigned int>(height),
      &sizes
   ));
   state_ = DeviceBuffer<unsigned char>(sizes.stateSizeInBytes);
   scratch_ = DeviceBuffer<unsigned char>(sizes.withoutOverlapScratchSizeInBytes);
   RT_OPTIX_CHECK(optixDenoiserSetup(
      denoiser_,
      nullptr,
      static_cast<unsigned int>(width),
      static_cast<unsigned int>(height),
      device_pointer(state_.data()),
      state_.size(),
      device_pointer(scratch_.data()),
      scratch_.size()
   ));
}

Denoiser::~Denoiser() {
   if (denoiser_ != nullptr && optixDenoiserDestroy(denoiser_) != OPTIX_SUCCESS) {
      std::cerr << "OptiX error during optixDenoiserDestroy\n";
   }
   if (context_ != nullptr && optixDeviceContextDestroy(context_) != OPTIX_SUCCESS) {
      std::cerr << "OptiX error during optixDeviceContextDestroy\n";
   }
}

OptixImage2D Denoiser::image(const rt::Vec3* pixels) const {
   return {
      .data = device_pointer(pixels),
      .width = static_cast<unsigned int>(width_),
      .height = static_cast<unsigned int>(height_),
      .rowStrideInBytes = static_cast<unsigned int>(width_ * sizeof(rt::Vec3)),
      .pixelStrideInBytes = sizeof(rt::Vec3),
      .format = OPTIX_PIXEL_FORMAT_FLOAT3,
   };
}

void Denoiser::denoise(
   const rt::Vec3* color,
   const rt::Vec3* albedo,
   const rt::Vec3* normal,
   rt::Vec3* output
) {
   const OptixImage2D input = image(color);
   RT_OPTIX_CHECK(optixDenoiserComputeIntensity(
      denoiser_,
      nullptr,
      &input,
      device_pointer(intensity_.data()),
      device_pointer(scratch_.data()),
      scratch_.size()
   ));
   RT_OPTIX_CHECK(optixDenoiserComputeAverageColor(
      denoiser_,
      nullptr,
      &input,
      device_pointer(average_color_.data()),
      device_pointer(scratch_.data()),
      scratch_.size()
   ));

   OptixDenoiserParams params{};
   params.hdrIntensity = device_pointer(intensity_.data());
   params.hdrAverageColor = device_pointer(average_color_.data());
   params.blendFactor = 0.0f;

   OptixDenoiserGuideLayer guide{};
   guide.albedo = image(albedo);
   guide.normal = image(normal);

   OptixDenoiserLayer layer{};
   layer.input = input;
   layer.output = image(output);

   RT_OPTIX_CHECK(optixDenoiserInvoke(
      denoiser_,
      nullptr,
      &params,
      device_pointer(state_.data()),
      state_.size(),
      &guide,
      &layer,
      1,
      0,
      0,
      device_pointer(scratch_.data()),
      scratch_.size()
   ));
}
