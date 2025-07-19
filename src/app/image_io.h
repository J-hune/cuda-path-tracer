#pragma once
#include "core/vec3.h"

#include <cstdint>
#include <string>
#include <vector>

// Convertit une image linéaire en octets RGBA (écrêtage dans [0, 1] puis gamma 2,2)
std::vector<std::uint8_t> to_rgba8(const std::vector<rt::Vec3>& image);

bool write_png(const std::string& path, const std::vector<rt::Vec3>& image, int width, int height);
