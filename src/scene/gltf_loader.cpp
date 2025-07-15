#include "scene/gltf_loader.h"

#include "core/camera.h"

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
#include <variant>

namespace rt {
std::expected<Scene, std::string> load_gltf(const std::filesystem::path& path) {
   if (!std::filesystem::exists(path)) {
      return std::unexpected("GLTF file not found: " + path.string());
   }

   fastgltf::Parser parser;
   Scene scene;

   auto data = fastgltf::GltfDataBuffer::FromPath(path);
   if (data.error() != fastgltf::Error::None) {
      return std::unexpected(
         "Failed to load GLTF file " + path.string() + ": " +
         std::string(fastgltf::getErrorMessage(data.error()))
      );
   }

   auto asset = parser.loadGltf(
      data.get(),
      path.parent_path(),
      fastgltf::Options::LoadExternalBuffers
   );
   if (auto error = asset.error(); error != fastgltf::Error::None) {
      return std::unexpected(
         "Failed to parse GLTF file " + path.string() + ": " +
         std::string(fastgltf::getErrorMessage(error))
      );
   }

   if (asset->scenes.empty()) {
      return std::unexpected("GLTF file " + path.string() + " contains no scene");
   }

   const std::size_t scene_index = asset->defaultScene.value_or(0);
   std::string error;

   fastgltf::iterateSceneNodes(
      asset.get(),
      scene_index,
      fastgltf::math::fmat4x4(),
      [&](fastgltf::Node& node, fastgltf::math::fmat4x4 matrix) {
         if (!error.empty()) {
            return;
         }

         if (node.cameraIndex.has_value() && !scene.camera.has_value()) {
            // Seule la première caméra de la scène est utilisée
            const auto& camera = asset->cameras[*node.cameraIndex];
            const auto* perspective = std::get_if<fastgltf::Camera::Perspective>(&camera.camera);
            if (perspective == nullptr) {
               error = "Camera '" + std::string(camera.name) +
                       "': only perspective cameras are supported";
               return;
            }

            const auto position = matrix * fastgltf::math::fvec4(0.f, 0.f, 0.f, 1.f);
            const auto target = matrix * fastgltf::math::fvec4(0.f, 0.f, -1.f, 1.f);
            const auto up = matrix * fastgltf::math::fvec4(0.f, 1.f, 0.f, 0.f);
            scene.camera = CameraSettings{
               .position = {position.x(), position.y(), position.z()},
               .target = {target.x(), target.y(), target.z()},
               .up = {up.x(), up.y(), up.z()},
               .vertical_fov_degrees = perspective->yfov * 180.f / std::numbers::pi_v<float>
            };
         }

         if (!node.meshIndex.has_value()) {
            return;
         }

         const auto& mesh = asset->meshes[*node.meshIndex];
         const std::string mesh_name(mesh.name);

         for (const auto& primitive : mesh.primitives) {
            if (primitive.type != fastgltf::PrimitiveType::Triangles) {
               error = "Mesh '" + mesh_name + "': primitive is not made of triangles";
               return;
            }

            auto position_attr = primitive.findAttribute("POSITION");
            if (position_attr == primitive.attributes.end()) {
               error = "Mesh '" + mesh_name + "': primitive has no POSITION attribute";
               return;
            }

            const auto& pos_accessor = asset->accessors[position_attr->accessorIndex];
            if (!pos_accessor.bufferViewIndex.has_value()) {
               error = "Mesh '" + mesh_name + "': POSITION accessor has no buffer view";
               return;
            }

            const fastgltf::Accessor* idx_accessor = nullptr;
            if (primitive.indicesAccessor.has_value()) {
               idx_accessor = &asset->accessors[*primitive.indicesAccessor];
               if (!idx_accessor->bufferViewIndex.has_value()) {
                  error = "Mesh '" + mesh_name + "': index accessor has no buffer view";
                  return;
               }
            }

            const auto base = static_cast<std::uint32_t>(scene.positions.size());

            fastgltf::iterateAccessor<fastgltf::math::fvec3>(
               asset.get(),
               pos_accessor,
               [&](fastgltf::math::fvec3 position) {
                  const auto world = matrix * fastgltf::math::fvec4(
                                                 position.x(),
                                                 position.y(),
                                                 position.z(),
                                                 1.0f
                                              );
                  scene.positions.push_back({world.x(), world.y(), world.z()});
               }
            );

            if (idx_accessor != nullptr) {
               fastgltf::iterateAccessor<std::uint32_t>(
                  asset.get(),
                  *idx_accessor,
                  [&](std::uint32_t index) { scene.indices.push_back(base + index); }
               );
            } else {
               for (std::size_t i = 0; i < pos_accessor.count; ++i) {
                  scene.indices.push_back(base + static_cast<std::uint32_t>(i));
               }
            }
         }
      }
   );

   if (!error.empty()) {
      return std::unexpected(error);
   }

   return scene;
}
} // namespace rt
