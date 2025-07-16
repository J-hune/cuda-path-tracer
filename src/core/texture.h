#pragma once

#include "core/host_device.h"
#include "core/vec2.h"
#include "core/vec3.h"

#include <cmath>
#include <cstdint>

namespace rt {

// Image RGB linéaire, stockée à partir de texels[offset], ligne par ligne
struct Texture {
   std::uint32_t offset = 0;
   std::uint32_t width = 0;
   std::uint32_t height = 0;
};

// Lecture au plus proche, texture répétée au-delà de [0, 1]. v = 0 est la première ligne de l'image
RT_HOST_DEVICE inline Vec3 sample_texture(Texture texture, const Vec3* texels, Vec2 uv) {
   const float u = uv.x - std::floor(uv.x);
   const float v = uv.y - std::floor(uv.y);
   const auto x = static_cast<std::uint32_t>(u * static_cast<float>(texture.width));
   const auto y = static_cast<std::uint32_t>(v * static_cast<float>(texture.height));
   const std::uint32_t column = x < texture.width ? x : texture.width - 1;
   const std::uint32_t row = y < texture.height ? y : texture.height - 1;
   return texels[texture.offset + (row * texture.width) + column];
}

} // namespace rt
