#pragma once

#include "core/host_device.h"
#include "core/ray.h"
#include "core/sampling.h"
#include "core/vec2.h"
#include "core/vec3.h"

#include <cmath>
#include <numbers>
#include <type_traits>

namespace rt {
struct CameraSettings {
   Vec3 position = {0.f, 0.f, 0.f};
   Vec3 target = {0.f, 0.f, -1.f};
   Vec3 up = {0.f, 1.f, 0.f};
   float vertical_fov_degrees = 60.f;
   // Profondeur de champ (lentille mince), en unités de la scène. Rayon nul : sténopé, tout est net
   float aperture_radius = 0.f;
   float focus_distance = 1.f; // distance du plan net, le long de l'axe de visée
};

static constexpr float degrees_to_radians(float degrees) {
   return degrees * std::numbers::pi_v<float> / 180.f;
}

class Camera {
public:
   Camera(const CameraSettings& settings, int width, int height):
      position_(settings.position),
      forward_(normalize(settings.target - settings.position)),
      right_(normalize(cross(forward_, settings.up))),
      up_(cross(right_, forward_)),
      half_height_(std::tan(degrees_to_radians(settings.vertical_fov_degrees) * 0.5f)),
      half_width_(half_height_ * static_cast<float>(width) / static_cast<float>(height)),
      inv_width_(1.f / static_cast<float>(width)),
      inv_height_(1.f / static_cast<float>(height)),
      aperture_radius_(settings.aperture_radius),
      focus_distance_(settings.focus_distance) {}

   [[nodiscard]] RT_HOST_DEVICE bool has_aperture() const { return aperture_radius_ > 0.f; }

   // Rayon passant par le point (x, y) de l'image. Avec une ouverture, il part d'un point de la
   // lentille tiré avec lens (carré unité) et vise le point net correspondant, sur le plan de mise
   // au point : seuls les objets à cette distance restent nets.
   [[nodiscard]] RT_HOST_DEVICE Ray generate_ray(float x, float y, Vec2 lens = {0.5f, 0.5f}) const {
      const float sx = 2.f * x * inv_width_ - 1.f;
      const float sy = 1.f - 2.f * y * inv_height_;
      const Vec3 direction = normalize(
         forward_ + sx * half_width_ * right_ + sy * half_height_ * up_
      );
      if (!has_aperture()) {
         return {position_, direction};
      }
      const Vec3 focus = position_ + direction * (focus_distance_ / dot(direction, forward_));
      const Vec2 d = square_to_disk_concentric(lens) * aperture_radius_;
      const Vec3 origin = position_ + d.x * right_ + d.y * up_;
      return {origin, normalize(focus - origin)};
   }

   // Angle couvert par un pixel, en radians (au centre de l'image)
   [[nodiscard]] RT_HOST_DEVICE float pixel_angle() const {
      return 2.f * half_height_ * inv_height_;
   }

   // Direction exprimée dans le repère de la caméra : x vers la droite, y vers le haut, z vers
   // l'observateur
   [[nodiscard]] RT_HOST_DEVICE Vec3 to_camera_space(Vec3 world) const {
      return {dot(world, right_), dot(world, up_), -dot(world, forward_)};
   }

private:
   Vec3 position_;
   Vec3 forward_;
   Vec3 right_;
   Vec3 up_;
   float half_height_;
   float half_width_;
   float inv_width_;
   float inv_height_;
   float aperture_radius_;
   float focus_distance_;
};

static_assert(std::is_trivially_copyable_v<Camera>);

} // namespace rt
