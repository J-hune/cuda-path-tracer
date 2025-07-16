#pragma once

#include "core/host_device.h"

#include <cmath>

namespace rt {

// Conversion d'une composante sRGB (encodée gamma, dans [0, 1]) vers une valeur linéaire
RT_HOST_DEVICE inline float srgb_to_linear(float c) {
   return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

} // namespace rt
