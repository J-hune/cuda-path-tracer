#pragma once
#include "core/host_device.h"

#include <cmath>
#include <limits>
#include <numbers>

namespace rt {
struct Vec3 {
   float x = 0.0f;
   float y = 0.0f;
   float z = 0.0f;

   RT_HOST_DEVICE constexpr Vec3& operator+=(Vec3 other) {
      x += other.x;
      y += other.y;
      z += other.z;
      return *this;
   }

   RT_HOST_DEVICE constexpr Vec3& operator-=(Vec3 other) {
      x -= other.x;
      y -= other.y;
      z -= other.z;
      return *this;
   }

   RT_HOST_DEVICE constexpr Vec3& operator*=(float s) {
      x *= s;
      y *= s;
      z *= s;
      return *this;
   }

   RT_HOST_DEVICE constexpr Vec3& operator/=(float s) {
      const float inv_s = 1.0f / s;
      return *this *= inv_s;
   }

   RT_HOST_DEVICE [[nodiscard]] constexpr bool operator==(Vec3 other) const {
      return x == other.x && y == other.y && z == other.z;
   }
};

inline constexpr float kInfinity = std::numeric_limits<float>::infinity();

inline constexpr float kPi = std::numbers::pi_v<float>;

RT_HOST_DEVICE constexpr Vec3 operator+(Vec3 a, Vec3 b) {
   return {a.x + b.x, a.y + b.y, a.z + b.z};
}

RT_HOST_DEVICE constexpr Vec3 operator-(Vec3 a, Vec3 b) {
   return {a.x - b.x, a.y - b.y, a.z - b.z};
}

RT_HOST_DEVICE constexpr Vec3 operator-(Vec3 v) {
   return {-v.x, -v.y, -v.z};
}

RT_HOST_DEVICE constexpr Vec3 operator*(Vec3 v, float s) {
   return {v.x * s, v.y * s, v.z * s};
}

RT_HOST_DEVICE constexpr Vec3 operator*(float s, Vec3 v) {
   return v * s;
}

RT_HOST_DEVICE constexpr Vec3 operator/(Vec3 v, float s) {
   const float inv_s = 1.0f / s;
   return v * inv_s;
}

RT_HOST_DEVICE constexpr Vec3 operator*(Vec3 a, Vec3 b) {
   return {a.x * b.x, a.y * b.y, a.z * b.z};
}

RT_HOST_DEVICE constexpr float dot(Vec3 a, Vec3 b) {
   return a.x * b.x + a.y * b.y + a.z * b.z;
}

RT_HOST_DEVICE constexpr Vec3 cross(Vec3 a, Vec3 b) {
   return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

RT_HOST_DEVICE constexpr float length_squared(Vec3 v) {
   return dot(v, v);
}

RT_HOST_DEVICE inline float length(Vec3 v) {
   return std::sqrt(length_squared(v));
}

// Précondition : v non nul. Non vérifié pour des raisons de performance
RT_HOST_DEVICE inline Vec3 normalize(Vec3 v) {
   return v / length(v);
}

// Minimum et maximum composante par composante
RT_HOST_DEVICE constexpr Vec3 min(Vec3 a, Vec3 b) {
   return {a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z};
}

RT_HOST_DEVICE constexpr Vec3 max(Vec3 a, Vec3 b) {
   return {a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z};
}

RT_HOST_DEVICE constexpr float min_component(Vec3 v) {
   const float xy = v.x < v.y ? v.x : v.y;
   return xy < v.z ? xy : v.z;
}

RT_HOST_DEVICE constexpr float max_component(Vec3 v) {
   const float xy = v.x > v.y ? v.x : v.y;
   return xy > v.z ? xy : v.z;
}

static_assert(sizeof(Vec3) == 12);
} // namespace rt
