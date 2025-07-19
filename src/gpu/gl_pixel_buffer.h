#pragma once

#include <cuda_runtime.h>

struct cudaGraphicsResource;

// Pixel buffer OpenGL enregistré auprès de CUDA : un kernel peut y écrire directement.
// Le buffer doit être mappé pendant l'écriture CUDA, puis démappé avant qu'OpenGL ne le lise.
class GlPixelBuffer {
public:
   explicit GlPixelBuffer(unsigned int gl_buffer);
   ~GlPixelBuffer();
   GlPixelBuffer(const GlPixelBuffer&) = delete;
   GlPixelBuffer& operator=(const GlPixelBuffer&) = delete;
   GlPixelBuffer(GlPixelBuffer&&) = delete;
   GlPixelBuffer& operator=(GlPixelBuffer&&) = delete;

   // Accès au buffer pendant la durée de vie de l'objet : mappé à la construction, démappé à la
   // destruction
   class Mapping {
   public:
      explicit Mapping(cudaGraphicsResource* resource);
      ~Mapping();
      Mapping(const Mapping&) = delete;
      Mapping& operator=(const Mapping&) = delete;
      Mapping(Mapping&&) = delete;
      Mapping& operator=(Mapping&&) = delete;

      [[nodiscard]] uchar4* pixels() const noexcept { return pixels_; }

   private:
      cudaGraphicsResource* resource_;
      uchar4* pixels_ = nullptr;
   };

   [[nodiscard]] Mapping map() { return Mapping(resource_); }

private:
   cudaGraphicsResource* resource_ = nullptr;
};
