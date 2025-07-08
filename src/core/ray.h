#pragma once

#include "core/host_device.h"
#include "core/vec3.h"

namespace rt {
struct Ray {
   Vec3 origin;
   Vec3 direction;

   RT_HOST_DEVICE [[nodiscard]] constexpr Vec3 at(float t) const { return origin + t * direction; }
};
} // namespace rt
