#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "app/image_io.h"

#include "core/vec3.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stb_image_write.h>
#include <string>
#include <vector>

namespace {

std::uint8_t encode_channel(float linear) {
   const float encoded = std::pow(std::clamp(linear, 0.f, 1.f), 1.f / 2.2f);
   return static_cast<std::uint8_t>(std::lround(encoded * 255.f));
}

} // namespace

std::vector<std::uint8_t> to_rgba8(const std::vector<rt::Vec3>& image) {
   std::vector<std::uint8_t> pixels;
   pixels.reserve(image.size() * 4);
   for (const rt::Vec3& pixel : image) {
      pixels.push_back(encode_channel(pixel.x));
      pixels.push_back(encode_channel(pixel.y));
      pixels.push_back(encode_channel(pixel.z));
      pixels.push_back(255);
   }
   return pixels;
}

bool write_png(const std::string& path, const std::vector<rt::Vec3>& image, int width, int height) {
   const std::vector<std::uint8_t> pixels = to_rgba8(image);
   return stbi_write_png(path.c_str(), width, height, 4, pixels.data(), width * 4) != 0;
}
