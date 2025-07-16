#pragma once

#include "core/host_device.h"

namespace rt {

struct Vec2 {
   float x = 0.0f;
   float y = 0.0f;
};

RT_HOST_DEVICE constexpr Vec2 operator+(Vec2 a, Vec2 b) {
   return {a.x + b.x, a.y + b.y};
}

RT_HOST_DEVICE constexpr Vec2 operator-(Vec2 a, Vec2 b) {
   return {a.x - b.x, a.y - b.y};
}

RT_HOST_DEVICE constexpr Vec2 operator*(Vec2 v, float s) {
   return {v.x * s, v.y * s};
}

RT_HOST_DEVICE constexpr Vec2 operator*(float s, Vec2 v) {
   return v * s;
}

} // namespace rt
