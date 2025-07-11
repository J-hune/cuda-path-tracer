#pragma once

#include "core/vec3.h"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace rt {
struct Scene {
   std::vector<rt::Vec3> positions;
   std::vector<std::uint32_t> indices;
};

std::expected<Scene, std::string> load_gltf(const std::filesystem::path& path);
} // namespace rt
