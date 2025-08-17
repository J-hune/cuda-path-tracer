#pragma once
#include "core/host_device.h"
#include "core/vec2.h"
#include "core/vec3.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace rt {

// Lumière venant de l'infini, là où le rayon ne touche rien.
// Dégradé vertical de bottom (vers le bas) à top (vers le haut), uniforme si les deux sont égaux.
// Noir par défaut : glTF ne décrit pas d'environnement.
struct Environment {
   Vec3 bottom{0.0f, 0.0f, 0.0f};
   Vec3 top{0.0f, 0.0f, 0.0f};

   [[nodiscard]] RT_HOST_DEVICE constexpr Vec3 radiance(Vec3 direction) const {
      const float t = 0.5f * (direction.y + 1.f);
      return ((1.f - t) * bottom) + (t * top);
   }
};

struct EnvironmentSample {
   Vec3 direction;
   float pdf = 0.0f; // par angle solide
};

// HDRI en projection équirectangulaire, avec la convention de Blender (repère glTF, +y en haut) :
// u = 1/2 - φ / 2π avec φ l'angle horizontal de Blender mesuré depuis +x (le centre de l'image),
// v = θ / π avec θ l'angle à +y, la ligne 0 de l'image est en haut.
// Les fonctions de répartition servent à tirer les directions selon la luminance de l'image.
struct EnvironmentMap {
   const Vec3* texels = nullptr; // nullptr : pas d'HDRI
   std::uint32_t width = 0;
   std::uint32_t height = 0;
   float strength = 1.0f;
   float rotation = 0.0f; // radians, autour de la verticale, comme le Mapping de Blender (Z)
   const float* marginal_cdf = nullptr;    // height + 1 entrées : choix de la ligne
   const float* conditional_cdf = nullptr; // height × (width + 1) : choix de la colonne
   float average_luminance = 0.0f;         // pour estimer sa puissance

   [[nodiscard]] RT_HOST_DEVICE bool present() const { return texels != nullptr; }

   [[nodiscard]] RT_HOST_DEVICE Vec2 direction_to_uv(Vec3 d) const {
      // Angle dans le plan horizontal de Blender : x Blender = x glTF, y Blender = -z glTF
      const float phi = std::atan2(-d.z, d.x) + rotation;
      float u = 0.5f - (phi / (2.0f * kPi));
      u -= std::floor(u);
      const float v = std::acos(std::fmin(std::fmax(d.y, -1.0f), 1.0f)) / kPi;
      return {u, v};
   }

   [[nodiscard]] RT_HOST_DEVICE Vec3 uv_to_direction(Vec2 uv) const {
      const float phi = (2.0f * kPi * (0.5f - uv.x)) - rotation;
      const float theta = kPi * uv.y;
      const float s = std::sin(theta);
      return {s * std::cos(phi), std::cos(theta), -s * std::sin(phi)};
   }

   [[nodiscard]] RT_HOST_DEVICE Vec3 radiance(Vec3 direction) const {
      const Vec2 uv = direction_to_uv(direction);
      const auto x = static_cast<std::uint32_t>(uv.x * static_cast<float>(width));
      const auto y = static_cast<std::uint32_t>(uv.y * static_cast<float>(height));
      const std::uint32_t column = x < width ? x : width - 1;
      const std::uint32_t row = y < height ? y : height - 1;
      return texels[(row * width) + column] * strength;
   }

   // Densité, par angle solide, de tirer cette direction avec sample : densité du texel sur
   // l'image, ramenée à l'angle solide par 1 / (2π² sin θ)
   [[nodiscard]] RT_HOST_DEVICE float pdf(Vec3 direction) const {
      const Vec2 uv = direction_to_uv(direction);
      const auto x = static_cast<std::uint32_t>(uv.x * static_cast<float>(width));
      const auto y = static_cast<std::uint32_t>(uv.y * static_cast<float>(height));
      const std::uint32_t column = x < width ? x : width - 1;
      const std::uint32_t row = y < height ? y : height - 1;
      const float* conditional = conditional_cdf + (static_cast<std::size_t>(row) * (width + 1));
      const float p_row = marginal_cdf[row + 1] - marginal_cdf[row];
      const float p_column = conditional[column + 1] - conditional[column];
      const float sin_theta = std::sin(kPi * uv.y);
      if (sin_theta <= 0.0f) {
         return 0.0f;
      }
      const auto texels_count = static_cast<float>(width) * static_cast<float>(height);
      return p_row * p_column * texels_count / (2.0f * kPi * kPi * sin_theta);
   }

   // Tire une direction selon la luminance (u dans [0, 1)²)
   [[nodiscard]] RT_HOST_DEVICE EnvironmentSample sample(Vec2 u) const {
      const std::uint32_t row = search(marginal_cdf, height, u.x);
      const float* conditional = conditional_cdf + (static_cast<std::size_t>(row) * (width + 1));
      const std::uint32_t column = search(conditional, width, u.y);
      // Position uniforme dans le texel choisi, à partir de ce qui reste de u
      const float row_width = marginal_cdf[row + 1] - marginal_cdf[row];
      const float column_width = conditional[column + 1] - conditional[column];
      const float fx = (u.y - conditional[column]) / column_width;
      const float fy = (u.x - marginal_cdf[row]) / row_width;
      const Vec2 uv{
         (static_cast<float>(column) + std::fmin(std::fmax(fx, 0.0f), 1.0f)) /
            static_cast<float>(width),
         (static_cast<float>(row) + std::fmin(std::fmax(fy, 0.0f), 1.0f)) /
            static_cast<float>(height),
      };
      const float sin_theta = std::sin(kPi * uv.y);
      if (sin_theta <= 0.0f) {
         return {.direction = uv_to_direction(uv), .pdf = 0.0f};
      }
      const auto texels_count = static_cast<float>(width) * static_cast<float>(height);
      return {
         .direction = uv_to_direction(uv),
         .pdf = row_width * column_width * texels_count / (2.0f * kPi * kPi * sin_theta),
      };
   }

private:
   // Index k tel que cdf[k] <= u < cdf[k + 1], parmi count intervalles
   RT_HOST_DEVICE static std::uint32_t search(const float* cdf, std::uint32_t count, float u) {
      std::uint32_t low = 0;
      std::uint32_t high = count;
      while (high - low > 1) {
         const std::uint32_t middle = (low + high) / 2;
         if (u < cdf[middle]) {
            high = middle;
         } else {
            low = middle;
         }
      }
      return low;
   }
};

} // namespace rt
