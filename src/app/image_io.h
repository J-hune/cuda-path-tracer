#pragma once
#include "core/vec3.h"

#include <string>
#include <vector>

bool write_png(const std::string& path, const std::vector<rt::Vec3>& image, int width, int height);
