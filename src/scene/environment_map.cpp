#include "scene/environment_map.h"

#include "core/environment.h"
#include "core/vec3.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <numbers>
#include <stb_image.h>
#include <string>
#include <tinyexr.h>
#include <vector>

namespace rt {

namespace {

float luminance(Vec3 c) {
   return (0.2126f * c.x) + (0.7152f * c.y) + (0.0722f * c.z);
}

} // namespace

EnvironmentMap EnvironmentImage::view() const {
   if (empty()) {
      return {};
   }
   return {
      .texels = texels.data(),
      .width = width,
      .height = height,
      .strength = strength,
      .rotation = rotation,
      .marginal_cdf = marginal_cdf.data(),
      .conditional_cdf = conditional_cdf.data(),
      .average_luminance = average_luminance,
   };
}

void build_environment_distribution(EnvironmentImage& image) {
   const std::uint32_t w = image.width;
   const std::uint32_t h = image.height;
   image.conditional_cdf.assign(static_cast<std::size_t>(h) * (w + 1), 0.0f);
   image.marginal_cdf.assign(h + 1, 0.0f);
   std::vector<double> row_weights(h, 0.0);
   double total_luminance = 0.0;
   for (std::uint32_t row = 0; row < h; ++row) {
      const double sin_theta = std::sin(
         std::numbers::pi * (static_cast<double>(row) + 0.5) / static_cast<double>(h)
      );
      float* conditional = image.conditional_cdf.data() + (static_cast<std::size_t>(row) * (w + 1));
      double running = 0.0;
      std::vector<double> prefix(w + 1, 0.0);
      for (std::uint32_t column = 0; column < w; ++column) {
         const float l = luminance(image.texels[(static_cast<std::size_t>(row) * w) + column]);
         total_luminance += l * sin_theta;
         running += std::fmax(l, 0.0f) * sin_theta;
         prefix[column + 1] = running;
      }
      row_weights[row] = running;
      for (std::uint32_t column = 0; column <= w; ++column) {
         // Ligne noire : choix uniforme de la colonne
         conditional[column] = running > 0.0 ? static_cast<float>(prefix[column] / running)
                                             : static_cast<float>(column) / static_cast<float>(w);
      }
      conditional[w] = 1.0f;
   }
   double total = 0.0;
   for (const double weight : row_weights) {
      total += weight;
   }
   double running = 0.0;
   for (std::uint32_t row = 0; row < h; ++row) {
      running += total > 0.0 ? row_weights[row] / total : 1.0 / static_cast<double>(h);
      image.marginal_cdf[row + 1] = static_cast<float>(running);
   }
   image.marginal_cdf[h] = 1.0f;
   // Moyenne sur la sphère : somme des luminances × sin θ × aire d'un texel / 4π
   const double texel_area = (std::numbers::pi / h) * (2.0 * std::numbers::pi / w);
   image.average_luminance = static_cast<float>(
      total_luminance * texel_area / (4.0 * std::numbers::pi)
   );
}

std::expected<EnvironmentImage, std::string> load_environment(
   const std::filesystem::path& path,
   float strength,
   float rotation
) {
   EnvironmentImage image;
   image.strength = strength;
   image.rotation = rotation;
   const std::string name = path.string();
   image.source = name;
   int width = 0;
   int height = 0;
   if (path.extension() == ".exr" || path.extension() == ".EXR") {
      float* rgba = nullptr;
      const char* error = nullptr;
      if (LoadEXR(&rgba, &width, &height, name.c_str(), &error) != TINYEXR_SUCCESS) {
         const std::string message = "Failed to load " + name + ": " +
                                     (error != nullptr ? error : "?");
         FreeEXRErrorMessage(error);
         return std::unexpected(message);
      }
      image.texels.reserve(static_cast<std::size_t>(width) * height);
      for (std::size_t i = 0; i < static_cast<std::size_t>(width) * height; ++i) {
         image.texels.push_back({rgba[(4 * i) + 0], rgba[(4 * i) + 1], rgba[(4 * i) + 2]});
      }
      std::free(rgba);
   } else {
      int channels = 0;
      float* rgb = stbi_loadf(name.c_str(), &width, &height, &channels, 3);
      if (rgb == nullptr) {
         return std::unexpected("Failed to load " + name + ": " + stbi_failure_reason());
      }
      image.texels.reserve(static_cast<std::size_t>(width) * height);
      for (std::size_t i = 0; i < static_cast<std::size_t>(width) * height; ++i) {
         image.texels.push_back({rgb[(3 * i) + 0], rgb[(3 * i) + 1], rgb[(3 * i) + 2]});
      }
      stbi_image_free(rgb);
   }
   image.width = static_cast<std::uint32_t>(width);
   image.height = static_cast<std::uint32_t>(height);
   build_environment_distribution(image);
   return image;
}

} // namespace rt
