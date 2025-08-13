#include "app/fly_camera.h"

#include "core/camera.h"
#include "core/vec3.h"

#include <algorithm>
#include <cmath>

namespace {

// Tangage limité juste avant la verticale : au-delà, la droite de la caméra n'est plus définie
constexpr float kMaxPitch = 1.55f;
constexpr rt::Vec3 kWorldUp{0.0f, 1.0f, 0.0f};

} // namespace

// Lacet nul : la caméra regarde vers -z. Lacet positif : elle tourne vers +x
FlyCamera::FlyCamera(const rt::CameraSettings& initial):
   position_(initial.position),
   vertical_fov_degrees_(initial.vertical_fov_degrees),
   aperture_radius_(initial.aperture_radius),
   focus_distance_(initial.focus_distance) {
   const rt::Vec3 direction = rt::normalize(initial.target - initial.position);
   yaw_ = std::atan2(direction.x, -direction.z);
   pitch_ = std::asin(std::clamp(direction.y, -1.0f, 1.0f));
}

rt::Vec3 FlyCamera::forward() const {
   return {std::sin(yaw_) * std::cos(pitch_), std::sin(pitch_), -std::cos(yaw_) * std::cos(pitch_)};
}

void FlyCamera::move(rt::Vec3 local_offset) {
   const rt::Vec3 front = forward();
   const rt::Vec3 right = rt::normalize(rt::cross(front, kWorldUp));
   position_ += (right * local_offset.x) + (kWorldUp * local_offset.y) + (front * local_offset.z);
}

void FlyCamera::rotate(float delta_yaw, float delta_pitch) {
   yaw_ += delta_yaw;
   pitch_ = std::clamp(pitch_ + delta_pitch, -kMaxPitch, kMaxPitch);
}

rt::CameraSettings FlyCamera::settings() const {
   return {
      .position = position_,
      .target = position_ + forward(),
      .up = kWorldUp,
      .vertical_fov_degrees = vertical_fov_degrees_,
      .aperture_radius = aperture_radius_,
      .focus_distance = focus_distance_,
   };
}

void FlyCamera::set_lens(float aperture_radius, float focus_distance) {
   aperture_radius_ = aperture_radius;
   focus_distance_ = focus_distance;
}
