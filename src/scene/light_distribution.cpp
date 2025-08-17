#include "scene/light_distribution.h"

#include "core/aabb.h"
#include "core/light.h"
#include "core/material.h"
#include "core/texture.h"
#include "core/vec2.h"
#include "core/vec3.h"
#include "scene/environment_map.h"
#include "scene/gltf_loader.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rt {

namespace {


float luminance(Vec3 c) {
   return (0.2126f * c.x) + (0.7152f * c.y) + (0.0722f * c.z);
}

// Moyenne de la texture sur le triangle uv0 uv1 uv2 (coordonnées de texture) : texels dont le
// centre est dans le triangle, ou texel du centre si le triangle est plus petit qu'un texel
Vec3 average_over_triangle(
   const Scene& scene,
   const Texture& texture,
   Vec2 uv0,
   Vec2 uv1,
   Vec2 uv2
) {
   const auto w = static_cast<float>(texture.width);
   const auto h = static_cast<float>(texture.height);
   const Vec2 p0{uv0.x * w, uv0.y * h};
   const Vec2 p1{uv1.x * w, uv1.y * h};
   const Vec2 p2{uv2.x * w, uv2.y * h};
   const auto texel = [&](int x, int y) {
      const auto wrap = [](int i, std::uint32_t n) {
         const int m = static_cast<int>(n);
         return static_cast<std::uint32_t>(((i % m) + m) % m);
      };
      return scene.texels
         [texture.offset + (wrap(y, texture.height) * texture.width) + wrap(x, texture.width)];
   };
   const auto edge = [](Vec2 a, Vec2 b, Vec2 p) {
      return ((b.x - a.x) * (p.y - a.y)) - ((b.y - a.y) * (p.x - a.x));
   };
   const float area = edge(p0, p1, p2);

   const int x0 = static_cast<int>(std::floor(std::min({p0.x, p1.x, p2.x})));
   const int x1 = static_cast<int>(std::ceil(std::max({p0.x, p1.x, p2.x})));
   const int y0 = static_cast<int>(std::floor(std::min({p0.y, p1.y, p2.y})));
   const int y1 = static_cast<int>(std::ceil(std::max({p0.y, p1.y, p2.y})));
   Vec3 sum{0.0f, 0.0f, 0.0f};
   int count = 0;
   if (area != 0.0f) {
      for (int y = y0; y < y1; ++y) {
         for (int x = x0; x < x1; ++x) {
            const Vec2 p{static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f};
            const float a = edge(p1, p2, p) / area;
            const float b = edge(p2, p0, p) / area;
            const float c = edge(p0, p1, p) / area;
            if (a >= 0.0f && b >= 0.0f && c >= 0.0f) {
               sum += texel(x, y);
               ++count;
            }
         }
      }
   }
   if (count == 0) {
      const Vec2 center{(p0.x + p1.x + p2.x) / 3.0f, (p0.y + p1.y + p2.y) / 3.0f};
      return texel(static_cast<int>(std::floor(center.x)), static_cast<int>(std::floor(center.y)));
   }
   return sum / static_cast<float>(count);
}

// Puissance émise par un triangle : π × aire × émission moyenne, par face émettrice
float triangle_power(const Scene& scene, std::uint32_t triangle) {
   const Material& material = scene.materials[scene.material_indices[triangle]];
   const std::uint32_t i0 = scene.indices[(3 * triangle) + 0];
   const std::uint32_t i1 = scene.indices[(3 * triangle) + 1];
   const std::uint32_t i2 = scene.indices[(3 * triangle) + 2];
   const Vec3 v0 = scene.positions[i0];
   const float area = 0.5f * length(cross(scene.positions[i1] - v0, scene.positions[i2] - v0));
   Vec3 emission = material.emission;
   if (material.emissive_texture >= 0) {
      const Texture& texture = scene.textures[static_cast<std::size_t>(material.emissive_texture)];
      emission = emission *
                 average_over_triangle(scene, texture, scene.uvs[i0], scene.uvs[i1], scene.uvs[i2]);
   }
   const float sides = material.double_sided ? 2.0f : 1.0f;
   return sides * kPi * area * luminance(emission);
}

float punctual_power(const PunctualLight& light, float scene_radius) {
   const float intensity = luminance(light.intensity);
   switch (light.type) {
      case PunctualLight::Type::Point:
         return 4.0f * kPi * intensity;
      case PunctualLight::Type::Spot:
         // Cône de demi-angle moyen entre le cône intérieur et le cône extérieur
         return 2.0f * kPi * (1.0f - (0.5f * (light.cos_inner + light.cos_outer))) * intensity;
      case PunctualLight::Type::Directional:
         break;
   }
   return kPi * scene_radius * scene_radius * intensity;
}

} // namespace

LightDistribution light_distribution(const Scene& scene) {
   Aabb bounds;
   for (const Vec3& p : scene.positions) {
      bounds.grow(p);
   }
   const float scene_radius = scene.positions.empty() ? 1.0f
                                                      : 0.5f * length(bounds.max - bounds.min);

   std::vector<float> powers;
   powers.reserve(scene.emissive_triangles.size() + scene.punctual_lights.size());
   for (const std::uint32_t triangle : scene.emissive_triangles) {
      powers.push_back(triangle_power(scene, triangle));
   }
   for (const PunctualLight& light : scene.punctual_lights) {
      powers.push_back(punctual_power(light, scene_radius));
   }
   // HDRI de luminance moyenne L : elle éclaire la sphère englobant la scène avec un éclairement
   // π L, sur une aire 4π R²
   if (!scene.environment_image.empty()) {
      const EnvironmentImage& env = scene.environment_image;
      powers.push_back(
         4.0f * kPi * kPi * scene_radius * scene_radius * env.average_luminance * env.strength
      );
   }

   LightDistribution distribution;
   distribution.triangle_light.assign(scene.material_indices.size(), -1);
   for (std::size_t k = 0; k < scene.emissive_triangles.size(); ++k) {
      distribution.triangle_light[scene.emissive_triangles[k]] = static_cast<std::int32_t>(k);
   }

   double total = 0.0;
   for (const float power : powers) {
      total += power;
   }
   distribution.cdf.reserve(powers.size() + 1);
   distribution.cdf.push_back(0.0f);
   double running = 0.0;
   for (const float power : powers) {
      // Puissance totale nulle : choix uniforme
      running += total > 0.0 ? power / total : 1.0 / static_cast<double>(powers.size());
      distribution.cdf.push_back(static_cast<float>(running));
   }
   if (!powers.empty()) {
      distribution.cdf.back() = 1.0f;
   }
   return distribution;
}

} // namespace rt
