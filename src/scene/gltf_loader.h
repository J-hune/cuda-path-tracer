#pragma once

#include "core/camera.h"
#include "core/material.h"
#include "core/vec3.h"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace rt {
struct Scene {
   std::vector<Vec3> positions;
   std::vector<std::uint32_t> indices;
   std::optional<CameraSettings> camera;
   std::vector<Material> materials;
   std::vector<std::uint32_t> material_indices;
};

std::expected<Scene, std::string> load_gltf(const std::filesystem::path& path);
} // namespace rt
