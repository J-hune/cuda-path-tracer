#pragma once

#include "core/host_device.h"
#include "core/ray.h"
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
      inv_height_(1.f / static_cast<float>(height)) {}

   [[nodiscard]] RT_HOST_DEVICE Ray generate_ray(float x, float y) const {
      const float sx = 2.f * x * inv_width_ - 1.f;
      const float sy = 1.f - 2.f * y * inv_height_;
      const Vec3 direction = normalize(
         forward_ + sx * half_width_ * right_ + sy * half_height_ * up_
      );
      return {position_, direction};
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
};

static_assert(std::is_trivially_copyable_v<Camera>);

} // namespace rt
