#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "app/image_io.h"

#include "core/tonemap.h"
#include "core/vec3.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stb_image_write.h>
#include <string>
#include <vector>

namespace {

std::uint8_t to_byte(float c) {
   return static_cast<std::uint8_t>(std::lround(std::clamp(c, 0.f, 1.f) * 255.f));
}

} // namespace

std::vector<std::uint8_t> to_rgba8(const std::vector<rt::Vec3>& image) {
   std::vector<std::uint8_t> pixels;
   pixels.reserve(image.size() * 4);
   for (const rt::Vec3& pixel : image) {
      const rt::Vec3 c = rt::display_color(pixel, rt::kDefaultExposure);
      pixels.push_back(to_byte(c.x));
      pixels.push_back(to_byte(c.y));
      pixels.push_back(to_byte(c.z));
      pixels.push_back(255);
   }
   return pixels;
}

bool write_png(const std::string& path, const std::vector<rt::Vec3>& image, int width, int height) {
   const std::vector<std::uint8_t> pixels = to_rgba8(image);
   return stbi_write_png(path.c_str(), width, height, 4, pixels.data(), width * 4) != 0;
}
