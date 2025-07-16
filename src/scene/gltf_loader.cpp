#include "scene/gltf_loader.h"

#include "core/camera.h"
#include "core/material.h"
#include "core/vec3.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <fastgltf/core.hpp>
#include <fastgltf/math.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <filesystem>
#include <numbers>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace rt {
namespace {

using Matrix = fastgltf::math::fmat4x4;

Vec3 transform(const Matrix& matrix, float x, float y, float z, float w) {
   const auto result = matrix * fastgltf::math::fvec4(x, y, z, w);
   return {result.x(), result.y(), result.z()};
}

std::string describe(fastgltf::Error error) {
   return std::string(fastgltf::getErrorMessage(error));
}

std::expected<fastgltf::Asset, std::string> parse_file(const std::filesystem::path& path) {
   if (!std::filesystem::exists(path)) {
      return std::unexpected("GLTF file not found: " + path.string());
   }

   auto data = fastgltf::GltfDataBuffer::FromPath(path);
   if (data.error() != fastgltf::Error::None) {
      return std::unexpected(
         "Failed to load GLTF file " + path.string() + ": " + describe(data.error())
      );
   }

   fastgltf::Parser parser(fastgltf::Extensions::KHR_materials_emissive_strength);
   auto asset = parser.loadGltf(
      data.get(),
      path.parent_path(),
      fastgltf::Options::LoadExternalBuffers
   );
   if (asset.error() != fastgltf::Error::None) {
      return std::unexpected(
         "Failed to parse GLTF file " + path.string() + ": " + describe(asset.error())
      );
   }
   if (asset->scenes.empty()) {
      return std::unexpected("GLTF file " + path.string() + " contains no scene");
   }
   return std::move(asset.get());
}

// Matériaux du fichier, suivis d'un matériau par défaut pour les primitives qui n'en ont pas
std::vector<Material> load_materials(const fastgltf::Asset& asset) {
   std::vector<Material> materials;
   for (const auto& material : asset.materials) {
      const auto& base_color = material.pbrData.baseColorFactor;
      const auto& emissive = material.emissiveFactor;
      const float strength = material.emissiveStrength;
      materials.push_back({
         .albedo = {base_color.x(), base_color.y(), base_color.z()},
         .emission = {emissive.x() * strength, emissive.y() * strength, emissive.z() * strength},
      });
   }
   materials.push_back({.albedo = {0.8f, 0.8f, 0.8f}, .emission = {0.f, 0.f, 0.f}});
   return materials;
}

// Caméra glTF : regarde vers -z dans son repère local, haut selon +y
std::expected<CameraSettings, std::string> load_camera(
   const fastgltf::Camera& camera,
   const Matrix& matrix
) {
   const auto* perspective = std::get_if<fastgltf::Camera::Perspective>(&camera.camera);
   if (perspective == nullptr) {
      return std::unexpected(
         "Camera '" + std::string(camera.name) + "': only perspective cameras are supported"
      );
   }
   return CameraSettings{
      .position = transform(matrix, 0.f, 0.f, 0.f, 1.f),
      .target = transform(matrix, 0.f, 0.f, -1.f, 1.f),
      .up = transform(matrix, 0.f, 1.f, 0.f, 0.f),
      .vertical_fov_degrees = perspective->yfov * 180.f / std::numbers::pi_v<float>,
   };
}

// Ajoute les sommets (en coordonnées monde), les indices et les matériaux d'une primitive
std::expected<void, std::string> load_primitive(
   const fastgltf::Asset& asset,
   const fastgltf::Primitive& primitive,
   const Matrix& matrix,
   std::uint32_t default_material,
   Scene& scene
) {
   if (primitive.type != fastgltf::PrimitiveType::Triangles) {
      return std::unexpected("primitive is not made of triangles");
   }

   const auto position_attribute = primitive.findAttribute("POSITION");
   if (position_attribute == primitive.attributes.end()) {
      return std::unexpected("primitive has no POSITION attribute");
   }
   const auto& position_accessor = asset.accessors[position_attribute->accessorIndex];
   if (!position_accessor.bufferViewIndex.has_value()) {
      return std::unexpected("POSITION accessor has no buffer view");
   }

   const fastgltf::Accessor* index_accessor = nullptr;
   if (primitive.indicesAccessor.has_value()) {
      index_accessor = &asset.accessors[*primitive.indicesAccessor];
      if (!index_accessor->bufferViewIndex.has_value()) {
         return std::unexpected("index accessor has no buffer view");
      }
   }

   const auto base = static_cast<std::uint32_t>(scene.positions.size());
   const std::size_t first_index = scene.indices.size();

   fastgltf::iterateAccessor<fastgltf::math::fvec3>(
      asset,
      position_accessor,
      [&](fastgltf::math::fvec3 p) {
         scene.positions.push_back(transform(matrix, p.x(), p.y(), p.z(), 1.f));
      }
   );

   if (index_accessor != nullptr) {
      fastgltf::iterateAccessor<std::uint32_t>(asset, *index_accessor, [&](std::uint32_t index) {
         scene.indices.push_back(base + index);
      });
   } else {
      for (std::size_t i = 0; i < position_accessor.count; ++i) {
         scene.indices.push_back(base + static_cast<std::uint32_t>(i));
      }
   }

   const std::size_t index_count = scene.indices.size() - first_index;
   if (index_count % 3 != 0) {
      return std::unexpected("index count is not a multiple of 3");
   }

   const std::uint32_t material_index = primitive.materialIndex.has_value()
                                           ? static_cast<std::uint32_t>(*primitive.materialIndex)
                                           : default_material;
   scene.material_indices.insert(scene.material_indices.end(), index_count / 3, material_index);
   return {};
}

} // namespace

std::expected<Scene, std::string> load_gltf(const std::filesystem::path& path) {
   auto parsed = parse_file(path);
   if (!parsed) {
      return std::unexpected(parsed.error());
   }
   const fastgltf::Asset& asset = *parsed;

   Scene scene;
   scene.materials = load_materials(asset);
   const auto default_material = static_cast<std::uint32_t>(scene.materials.size() - 1);

   // iterateSceneNodes prend un lambda : on garde la première erreur pour la renvoyer ensuite
   std::string error;
   fastgltf::iterateSceneNodes(
      asset,
      asset.defaultScene.value_or(0),
      Matrix(),
      [&](const fastgltf::Node& node, const Matrix& matrix) {
         if (!error.empty()) {
            return;
         }

         // Seule la première caméra de la scène est utilisée
         if (node.cameraIndex.has_value() && !scene.camera.has_value()) {
            auto camera = load_camera(asset.cameras[*node.cameraIndex], matrix);
            if (!camera) {
               error = camera.error();
               return;
            }
            scene.camera = *camera;
         }

         if (!node.meshIndex.has_value()) {
            return;
         }
         const auto& mesh = asset.meshes[*node.meshIndex];
         for (const auto& primitive : mesh.primitives) {
            auto loaded = load_primitive(asset, primitive, matrix, default_material, scene);
            if (!loaded) {
               error = "Mesh '" + std::string(mesh.name) + "': " + loaded.error();
               return;
            }
         }
      }
   );

   if (!error.empty()) {
      return std::unexpected(error);
   }

   // Triangles émissifs, tirés comme sources de lumière par l'éclairage direct
   for (std::size_t triangle = 0; triangle < scene.material_indices.size(); ++triangle) {
      const Material& material = scene.materials[scene.material_indices[triangle]];
      if (max_component(material.emission) > 0.f) {
         scene.emissive_triangles.push_back(static_cast<std::uint32_t>(triangle));
      }
   }
   return scene;
}

} // namespace rt
