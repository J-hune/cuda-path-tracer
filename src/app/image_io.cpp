#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "app/image_io.h"

#include "core/vec3.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stb_image_write.h>
#include <string>
#include <vector>

bool write_png(const std::string& path, const std::vector<rt::Vec3>& image, int width, int height) {
   std::vector<unsigned char> pixels;
   pixels.reserve(static_cast<std::size_t>(width) * height * 3);

   for (const auto& pixel : image) {
      for (float channel : {pixel.x, pixel.y, pixel.z}) {
         channel = std::clamp(channel, 0.f, 1.f);
         channel = std::pow(channel, 1.f / 2.2f);

         pixels.push_back(static_cast<unsigned char>(std::lround(channel * 255.f)));
      }
   }

   return stbi_write_png(path.c_str(), width, height, 3, pixels.data(), width * 3) != 0;
}
