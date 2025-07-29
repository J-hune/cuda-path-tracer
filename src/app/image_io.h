#pragma once
#include "core/vec3.h"

#include <cstdint>
#include <string>
#include <vector>

// Convertit une image linéaire en octets RGBA affichables (tone mapping ACES puis sRGB)
std::vector<std::uint8_t> to_rgba8(const std::vector<rt::Vec3>& image);

// PNG 8 bits, après tone mapping (ce qu'affiche la fenêtre)
bool write_png(const std::string& path, const std::vector<rt::Vec3>& image, int width, int height);

// EXR linéaire en demi-précision : la radiance brute, sans tone mapping
bool write_exr(const std::string& path, const std::vector<rt::Vec3>& image, int width, int height);
