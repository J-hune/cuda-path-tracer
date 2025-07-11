#include "scene/gltf_loader.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <fastgltf/core.hpp>
#include <fastgltf/math.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <filesystem>
#include <string>

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

   auto asset =
      parser.loadGltf(data.get(), path.parent_path(), fastgltf::Options::LoadExternalBuffers);
   if (auto error = asset.error(); error != fastgltf::Error::None) {
      return std::unexpected(
         "Failed to parse GLTF file " + path.string() + ": " +
         std::string(fastgltf::getErrorMessage(error))
      );
   }

   for (const auto& mesh : asset->meshes) {
      const std::string mesh_name(mesh.name);

      for (const auto& primitive : mesh.primitives) {
         if (primitive.type != fastgltf::PrimitiveType::Triangles) {
            return std::unexpected("Mesh '" + mesh_name + "': primitive is not made of triangles");
         }

         auto position_attr = primitive.findAttribute("POSITION");
         if (position_attr == primitive.attributes.end()) {
            return std::unexpected("Mesh '" + mesh_name + "': primitive has no POSITION attribute");
         }

         const auto& pos_accessor = asset->accessors[position_attr->accessorIndex];
         if (!pos_accessor.bufferViewIndex.has_value()) {
            return std::unexpected(
               "Mesh '" + mesh_name + "': POSITION accessor has no buffer view"
            );
         }

         auto base = static_cast<std::uint32_t>(scene.positions.size());

         fastgltf::iterateAccessor<fastgltf::math::fvec3>(
            asset.get(), pos_accessor, [&](fastgltf::math::fvec3 position) {
               scene.positions.push_back({position.x(), position.y(), position.z()});
            }
         );

         if (primitive.indicesAccessor.has_value()) {
            const auto& idx_accessor = asset->accessors[*primitive.indicesAccessor];
            if (!idx_accessor.bufferViewIndex.has_value()) {
               return std::unexpected(
                  "Mesh '" + mesh_name + "': index accessor has no buffer view"
               );
            }

            fastgltf::iterateAccessor<std::uint32_t>(
               asset.get(), idx_accessor, [&](std::uint32_t index) {
                  scene.indices.push_back(base + index);
               }
            );
         } else {
            for (std::size_t i = 0; i < pos_accessor.count; ++i) {
               scene.indices.push_back(base + static_cast<std::uint32_t>(i));
            }
         }
      }
   }

   return scene;
}
} // namespace rt
