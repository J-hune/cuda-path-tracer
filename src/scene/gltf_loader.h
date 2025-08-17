#pragma once

#include "core/camera.h"
#include "core/environment.h"
#include "core/light.h"
#include "core/material.h"
#include "core/tangent.h"
#include "core/texture.h"
#include "core/vec2.h"
#include "core/vec3.h"
#include "scene/environment_map.h"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace rt {
struct Scene {
   std::vector<Vec3> positions;
   std::vector<Vec2> uvs;
   std::vector<Vec3> normals; // par sommet ; nulle : ombrage plat
   std::vector<Tangent> tangents;
   std::vector<std::uint32_t> indices;
   std::optional<CameraSettings> camera;
   std::vector<Material> materials;
   std::vector<std::uint32_t> material_indices;
   std::vector<std::uint32_t> emissive_triangles;
   std::vector<PunctualLight> punctual_lights;
   std::vector<Texture> textures;
   std::vector<Vec3> texels;
   Environment environment;
   EnvironmentImage environment_image; // HDRI, vide si aucune
};

std::expected<Scene, std::string> load_gltf(const std::filesystem::path& path);
} // namespace rt
