#pragma once

#include "core/vec3.h"
#include "gpu/device_buffer.h"

#include <optix_types.h>

// Débruiteur d'OptiX, guidé par l'albédo et les normales.
// Images RGB float sur le GPU, en radiance linéaire moyenne
class Denoiser {
public:
   Denoiser(int width, int height);
   ~Denoiser();
   Denoiser(const Denoiser&) = delete;
   Denoiser& operator=(const Denoiser&) = delete;
   Denoiser(Denoiser&&) = delete;
   Denoiser& operator=(Denoiser&&) = delete;

   // Normales dans le repère de la caméra
   void denoise(
      const rt::Vec3* color,
      const rt::Vec3* albedo,
      const rt::Vec3* normal,
      rt::Vec3* output
   );

private:
   [[nodiscard]] OptixImage2D image(const rt::Vec3* pixels) const;

   int width_;
   int height_;
   OptixDeviceContext context_ = nullptr;
   OptixDenoiser denoiser_ = nullptr;
   DeviceBuffer<unsigned char> state_{0};
   DeviceBuffer<unsigned char> scratch_{0};
   DeviceBuffer<float> intensity_{1};
   DeviceBuffer<float> average_color_{3};
};
