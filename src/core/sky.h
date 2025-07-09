#pragma once
#include "core/host_device.h"
#include "core/vec3.h"

namespace rt {
RT_HOST_DEVICE constexpr Vec3 sky_color(Vec3 direction) {
   const float t = 0.5f * (direction.y + 1.f);
   return (1.f - t) * Vec3{1.0f, 1.0f, 1.0f} + t * Vec3{0.5f, 0.7f, 1.0f};
}
} // namespace rt
