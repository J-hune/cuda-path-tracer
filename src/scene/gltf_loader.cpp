#include "scene/gltf_loader.h"

#include "core/camera.h"
#include "core/color.h"
#include "core/light.h"
#include "core/material.h"
#include "core/vec2.h"
#include "core/vec3.h"
#include "scene/environment_map.h"
#include "scene/tangents.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <fastgltf/core.hpp>
#include <fastgltf/math.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <filesystem>
#include <map>
#include <numbers>
#include <simdjson.h>
#include <span>
#include <stb_image.h>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
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

// Normale transformée par l'inverse transposée de la partie 3 × 3 de la matrice, calculée avec la
// matrice des cofacteurs (colonnes c1 × c2, c2 × c0, c0 × c1), qui vaut det · (M⁻¹)ᵀ
Vec3 transform_normal(const Matrix& matrix, Vec3 n) {
   const Vec3 c0 = transform(matrix, 1.f, 0.f, 0.f, 0.f);
   const Vec3 c1 = transform(matrix, 0.f, 1.f, 0.f, 0.f);
   const Vec3 c2 = transform(matrix, 0.f, 0.f, 1.f, 0.f);
   const Vec3 transformed = (cross(c1, c2) * n.x) + (cross(c2, c0) * n.y) + (cross(c0, c1) * n.z);
   const float determinant = dot(c0, cross(c1, c2));
   return normalize(determinant < 0.0f ? -transformed : transformed);
}

std::string describe(fastgltf::Error error) {
   return std::string(fastgltf::getErrorMessage(error));
}

// Profondeur de champ des caméras, que glTF ne décrit pas : propriétés personnalisées (extras)
// raytracer_aperture (rayon de la lentille) et raytracer_focus_distance
struct CameraLens {
   float aperture_radius = 0.0f;
   float focus_distance = 1.0f;
};
using CameraLenses = std::unordered_map<std::size_t, CameraLens>;

// HDRI de la scène, que glTF ne décrit pas non plus : propriétés personnalisées de la scène
// raytracer_environment (chemin, relatif au fichier glTF), raytracer_environment_strength et
// raytracer_environment_rotation (degrés, autour de l'axe vertical)
struct SceneEnvironment {
   std::string path;
   float strength = 1.0f;
   float rotation_degrees = 0.0f;
};

struct Extras {
   CameraLenses lenses;
   SceneEnvironment environment;
};

void parse_scene_extras(simdjson::dom::object* extras, SceneEnvironment& environment) {
   std::string_view path;
   if ((*extras)["raytracer_environment"].get_string().get(path) == simdjson::SUCCESS) {
      environment.path = std::string(path);
   }
   double value = 0.0;
   if ((*extras)["raytracer_environment_strength"].get_double().get(value) == simdjson::SUCCESS) {
      environment.strength = static_cast<float>(value);
   }
   if ((*extras)["raytracer_environment_rotation"].get_double().get(value) == simdjson::SUCCESS) {
      environment.rotation_degrees = static_cast<float>(value);
   }
}

void parse_extras(
   simdjson::dom::object* extras,
   std::size_t index,
   fastgltf::Category category,
   void* user_pointer
) {
   auto& parsed = *static_cast<Extras*>(user_pointer);
   if (category == fastgltf::Category::Scenes) {
      parse_scene_extras(extras, parsed.environment);
      return;
   }
   if (category != fastgltf::Category::Cameras) {
      return;
   }
   CameraLens lens;
   double value = 0.0;
   if ((*extras)["raytracer_aperture"].get_double().get(value) == simdjson::SUCCESS) {
      lens.aperture_radius = static_cast<float>(value);
   }
   if ((*extras)["raytracer_focus_distance"].get_double().get(value) == simdjson::SUCCESS) {
      lens.focus_distance = static_cast<float>(value);
   }
   parsed.lenses[index] = lens;
}

std::expected<fastgltf::Asset, std::string> parse_file(
   const std::filesystem::path& path,
   Extras& extras
) {
   if (!std::filesystem::exists(path)) {
      return std::unexpected("GLTF file not found: " + path.string());
   }

   auto data = fastgltf::GltfDataBuffer::FromPath(path);
   if (data.error() != fastgltf::Error::None) {
      return std::unexpected(
         "Failed to load GLTF file " + path.string() + ": " + describe(data.error())
      );
   }

   constexpr auto kExtensions = fastgltf::Extensions::KHR_materials_emissive_strength |
                                fastgltf::Extensions::KHR_materials_transmission |
                                fastgltf::Extensions::KHR_materials_ior |
                                fastgltf::Extensions::KHR_materials_volume |
                                fastgltf::Extensions::KHR_materials_dispersion |
                                fastgltf::Extensions::KHR_lights_punctual;
   fastgltf::Parser parser(kExtensions);
   parser.setUserPointer(&extras);
   parser.setExtrasParseCallback(parse_extras);
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

// Octets encodés (PNG, JPEG…) d'une image, qu'elle soit dans un buffer ou intégrée directement
std::expected<std::span<const std::byte>, std::string> image_bytes(
   const fastgltf::Asset& asset,
   const fastgltf::Image& image
) {
   const auto bytes_of =
      [](const auto& source) -> std::expected<std::span<const std::byte>, std::string> {
      using Source = std::decay_t<decltype(source)>;
      if constexpr (std::is_same_v<Source, fastgltf::sources::Array> ||
                    std::is_same_v<Source, fastgltf::sources::Vector>) {
         return std::span<const std::byte>(source.bytes.data(), source.bytes.size());
      } else if constexpr (std::is_same_v<Source, fastgltf::sources::ByteView>) {
         return std::span<const std::byte>(source.bytes);
      } else {
         return std::unexpected(std::string("unsupported image data source"));
      }
   };

   if (const auto* view = std::get_if<fastgltf::sources::BufferView>(&image.data)) {
      const auto& buffer_view = asset.bufferViews[view->bufferViewIndex];
      const auto buffer = std::visit(bytes_of, asset.buffers[buffer_view.bufferIndex].data);
      if (!buffer) {
         return std::unexpected(buffer.error());
      }
      return buffer->subspan(buffer_view.byteOffset, buffer_view.byteLength);
   }
   return std::visit(bytes_of, image.data);
}

// Décode toutes les textures en RGB linéaire, mises bout à bout dans scene.texels
// Décode la texture glTF texture_index et l'ajoute à la scène. Les textures de couleur (couleur de
// base, émission) sont en sRGB ; les textures de données (metallic-roughness, normales) sont déjà
// linéaires.
std::expected<std::int32_t, std::string> load_texture(
   const fastgltf::Asset& asset,
   std::size_t texture_index,
   bool srgb,
   Scene& scene
) {
   const auto& texture = asset.textures[texture_index];
   if (!texture.imageIndex.has_value()) {
      return std::unexpected("Texture '" + std::string(texture.name) + "' has no image");
   }
   const auto& image = asset.images[*texture.imageIndex];
   const std::string image_name(image.name);

   const auto bytes = image_bytes(asset, image);
   if (!bytes) {
      return std::unexpected("Image '" + image_name + "': " + bytes.error());
   }

   int width = 0;
   int height = 0;
   int channels = 0;
   stbi_uc* pixels = stbi_load_from_memory(
      reinterpret_cast<const stbi_uc*>(bytes->data()),
      static_cast<int>(bytes->size()),
      &width,
      &height,
      &channels,
      3
   );
   if (pixels == nullptr) {
      return std::unexpected("Image '" + image_name + "': " + stbi_failure_reason());
   }

   scene.textures.push_back({
      .offset = static_cast<std::uint32_t>(scene.texels.size()),
      .width = static_cast<std::uint32_t>(width),
      .height = static_cast<std::uint32_t>(height),
   });
   const std::size_t pixel_count = static_cast<std::size_t>(width) * height;
   for (std::size_t i = 0; i < pixel_count; ++i) {
      const auto channel = [&](std::size_t c) {
         const float value = static_cast<float>(pixels[(3 * i) + c]) / 255.f;
         return srgb ? srgb_to_linear(value) : value;
      };
      scene.texels.push_back({channel(0), channel(1), channel(2)});
   }
   stbi_image_free(pixels);
   return static_cast<std::int32_t>(scene.textures.size() - 1);
}

// Textures chargées à la demande, une seule fois par usage (couleur ou données)
class TextureCache {
public:
   TextureCache(const fastgltf::Asset& asset, Scene& scene):
      asset_(asset),
      scene_(scene) {}

   // Index de la texture dans la scène, -1 sans texture
   // info : fastgltf::Optional d'une information de texture (TextureInfo ou dérivée)
   template <typename OptionalInfo>
   std::expected<std::int32_t, std::string> get(
      const OptionalInfo& info,
      bool srgb,
      std::string_view material
   ) {
      if (!info.has_value()) {
         return -1;
      }
      if (info->texCoordIndex != 0) {
         return std::unexpected(
            "Material '" + std::string(material) + "': only TEXCOORD_0 is supported"
         );
      }
      const auto key = std::pair{info->textureIndex, srgb};
      if (const auto found = loaded_.find(key); found != loaded_.end()) {
         return found->second;
      }
      auto index = load_texture(asset_, info->textureIndex, srgb, scene_);
      if (index) {
         loaded_[key] = *index;
      }
      return index;
   }

private:
   const fastgltf::Asset& asset_;
   Scene& scene_;
   std::map<std::pair<std::size_t, bool>, std::int32_t> loaded_;
};

std::expected<std::vector<Material>, std::string> load_materials(
   const fastgltf::Asset& asset,
   Scene& scene
) {
   TextureCache textures(asset, scene);
   std::vector<Material> materials;
   for (const auto& material : asset.materials) {
      const auto& base_color = material.pbrData.baseColorFactor;
      const auto& emissive = material.emissiveFactor;
      const float strength = material.emissiveStrength;

      const auto base_color_texture = textures.get(
         material.pbrData.baseColorTexture,
         true,
         material.name
      );
      const auto metallic_roughness_texture = textures.get(
         material.pbrData.metallicRoughnessTexture,
         false,
         material.name
      );
      const auto emissive_texture = textures.get(material.emissiveTexture, true, material.name);
      const auto normal_texture = textures.get(material.normalTexture, false, material.name);
      for (const auto* texture :
           {&base_color_texture, &metallic_roughness_texture, &emissive_texture, &normal_texture}) {
         if (!*texture) {
            return std::unexpected(texture->error());
         }
      }

      const float transmission = material.transmission
                                    ? static_cast<float>(material.transmission->transmissionFactor)
                                    : 0.0f;

      // KHR_materials_volume : couleur atteinte après attenuationDistance -> coefficient
      // d'absorption
      Vec3 absorption{0.0f, 0.0f, 0.0f};
      if (material.volume && material.volume->thicknessFactor > 0.0f &&
          std::isfinite(material.volume->attenuationDistance)) {
         const auto& color = material.volume->attenuationColor;
         const float distance = material.volume->attenuationDistance;
         absorption = {
            -std::log(color.x()) / distance,
            -std::log(color.y()) / distance,
            -std::log(color.z()) / distance
         };
      }

      materials.push_back({
         .albedo = {base_color.x(), base_color.y(), base_color.z()},
         .emission = {emissive.x() * strength, emissive.y() * strength, emissive.z() * strength},
         .base_color_texture = *base_color_texture,
         .metallic_roughness_texture = *metallic_roughness_texture,
         .emissive_texture = *emissive_texture,
         .normal_texture = *normal_texture,
         .normal_scale = material.normalTexture.has_value()
                            ? static_cast<float>(material.normalTexture->scale)
                            : 1.0f,
         .metallic = static_cast<float>(material.pbrData.metallicFactor),
         .roughness = static_cast<float>(material.pbrData.roughnessFactor),
         .transmission = transmission,
         .ior = material.ior,
         .absorption = absorption,
         .dispersion = static_cast<float>(material.dispersion),
         .double_sided = material.doubleSided,
      });
   }
   materials.push_back({.albedo = {0.8f, 0.8f, 0.8f}, .emission = {0.f, 0.f, 0.f}});
   return materials;
}

// Caméra glTF : regarde vers -z dans son repère local, haut selon +y
std::expected<CameraSettings, std::string> load_camera(
   const fastgltf::Camera& camera,
   const Matrix& matrix,
   CameraLens lens
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
      .aperture_radius = lens.aperture_radius,
      .focus_distance = lens.focus_distance,
   };
}

// Accesseur d'un attribut de sommet, vérifié ; nullptr si la primitive n'a pas cet attribut
std::expected<const fastgltf::Accessor*, std::string> find_attribute(
   const fastgltf::Asset& asset,
   const fastgltf::Primitive& primitive,
   std::string_view name
) {
   const auto attribute = primitive.findAttribute(name);
   if (attribute == primitive.attributes.end()) {
      return nullptr;
   }
   const auto& accessor = asset.accessors[attribute->accessorIndex];
   if (!accessor.bufferViewIndex.has_value()) {
      return std::unexpected(std::string(name) + " accessor has no buffer view");
   }
   return &accessor;
}

// Lumière ponctuelle glTF. Les intensités glTF sont photométriques (candela, lux) ; le moteur
// travaille en unités radiométriques (W/sr, W/m²), avec l'équivalence standard de 683 lm/W
PunctualLight load_light(const fastgltf::Light& light, const Matrix& matrix) {
   constexpr float kLumensPerWatt = 683.0f;
   const float intensity = static_cast<float>(light.intensity) / kLumensPerWatt;
   PunctualLight result{
      .position = transform(matrix, 0.f, 0.f, 0.f, 1.f),
      .direction = normalize(transform(matrix, 0.f, 0.f, -1.f, 0.f)),
      .intensity = Vec3{light.color.x(), light.color.y(), light.color.z()} * intensity,
   };
   switch (light.type) {
      case fastgltf::LightType::Point:
         result.type = PunctualLight::Type::Point;
         break;
      case fastgltf::LightType::Directional:
         result.type = PunctualLight::Type::Directional;
         break;
      case fastgltf::LightType::Spot:
         // Valeurs par défaut de glTF : cône intérieur de 0, cône extérieur de π/4
         result.type = PunctualLight::Type::Spot;
         result.cos_inner = std::cos(static_cast<float>(light.innerConeAngle.value_or(0.0f)));
         result.cos_outer = std::cos(
            static_cast<float>(light.outerConeAngle.value_or(std::numbers::pi_v<float> / 4))
         );
         break;
   }
   return result;
}

// Ajoute les sommets (en coordonnées monde), les UV, les indices et les matériaux d'une primitive

std::expected<void, std::string> load_primitive(
   const fastgltf::Asset& asset,
   const fastgltf::Primitive& primitive,
   const Matrix& matrix,
   std::uint32_t default_material,
   Scene& scene
) {
   // 1. Vérifications : rien n'est ajouté à la scène si la primitive est invalide
   if (primitive.type != fastgltf::PrimitiveType::Triangles) {
      return std::unexpected("primitive is not made of triangles");
   }

   const auto positions = find_attribute(asset, primitive, "POSITION");
   if (!positions) {
      return std::unexpected(positions.error());
   }
   if (*positions == nullptr) {
      return std::unexpected("primitive has no POSITION attribute");
   }
   const std::size_t vertex_count = (*positions)->count;

   const auto uvs = find_attribute(asset, primitive, "TEXCOORD_0");
   if (!uvs) {
      return std::unexpected(uvs.error());
   }
   if (*uvs != nullptr && (*uvs)->count != vertex_count) {
      return std::unexpected("TEXCOORD_0 and POSITION have different counts");
   }

   const auto normals = find_attribute(asset, primitive, "NORMAL");
   if (!normals) {
      return std::unexpected(normals.error());
   }
   if (*normals != nullptr && (*normals)->count != vertex_count) {
      return std::unexpected("NORMAL and POSITION have different counts");
   }

   const auto tangents = find_attribute(asset, primitive, "TANGENT");
   if (!tangents) {
      return std::unexpected(tangents.error());
   }
   if (*tangents != nullptr && (*tangents)->count != vertex_count) {
      return std::unexpected("TANGENT and POSITION have different counts");
   }

   const fastgltf::Accessor* indices = nullptr;
   if (primitive.indicesAccessor.has_value()) {
      indices = &asset.accessors[*primitive.indicesAccessor];
      if (!indices->bufferViewIndex.has_value()) {
         return std::unexpected("index accessor has no buffer view");
      }
   }
   const std::size_t index_count = indices != nullptr ? indices->count : vertex_count;
   if (index_count % 3 != 0) {
      return std::unexpected("index count is not a multiple of 3");
   }

   // 2. Ajout à la scène
   const auto base = static_cast<std::uint32_t>(scene.positions.size());

   fastgltf::iterateAccessor<fastgltf::math::fvec3>(
      asset,
      **positions,
      [&](fastgltf::math::fvec3 p) {
         scene.positions.push_back(transform(matrix, p.x(), p.y(), p.z(), 1.f));
      }
   );

   // Sans normales, glTF impose un ombrage plat : normale nulle, la normale géométrique sera
   // utilisée
   if (*normals != nullptr) {
      fastgltf::iterateAccessor<fastgltf::math::fvec3>(
         asset,
         **normals,
         [&](fastgltf::math::fvec3 n) {
            scene.normals.push_back(transform_normal(matrix, {n.x(), n.y(), n.z()}));
         }
      );
   } else {
      scene.normals.insert(scene.normals.end(), vertex_count, Vec3{0.f, 0.f, 0.f});
   }

   if (*uvs != nullptr) {
      fastgltf::iterateAccessor<fastgltf::math::fvec2>(asset, **uvs, [&](fastgltf::math::fvec2 uv) {
         scene.uvs.push_back({uv.x(), uv.y()});
      });
   } else {
      scene.uvs.insert(scene.uvs.end(), vertex_count, Vec2{0.f, 0.f});
   }

   if (indices != nullptr) {
      fastgltf::iterateAccessor<std::uint32_t>(asset, *indices, [&](std::uint32_t index) {
         scene.indices.push_back(base + index);
      });
   } else {
      for (std::size_t i = 0; i < vertex_count; ++i) {
         scene.indices.push_back(base + static_cast<std::uint32_t>(i));
      }
   }

   // Tangentes : celles du fichier, transformées comme des directions de la surface (une symétrie
   // inverse le sens de la bitangente), sinon calculées à partir des coordonnées de texture
   if (*tangents != nullptr) {
      const Vec3 c0 = transform(matrix, 1.f, 0.f, 0.f, 0.f);
      const Vec3 c1 = transform(matrix, 0.f, 1.f, 0.f, 0.f);
      const Vec3 c2 = transform(matrix, 0.f, 0.f, 1.f, 0.f);
      const float handedness = dot(c0, cross(c1, c2)) < 0.0f ? -1.0f : 1.0f;
      fastgltf::iterateAccessor<fastgltf::math::fvec4>(
         asset,
         **tangents,
         [&](fastgltf::math::fvec4 t) {
            const Vec3 direction = transform(matrix, t.x(), t.y(), t.z(), 0.f);
            scene.tangents.push_back({
               .direction = length_squared(direction) > 0.0f ? normalize(direction) : direction,
               .sign = (t.w() < 0.0f ? -1.0f : 1.0f) * handedness,
            });
         }
      );
   } else {
      compute_tangents(scene, base, static_cast<std::uint32_t>(vertex_count), index_count);
   }

   const std::uint32_t material_index = primitive.materialIndex.has_value()
                                           ? static_cast<std::uint32_t>(*primitive.materialIndex)
                                           : default_material;
   scene.material_indices.insert(scene.material_indices.end(), index_count / 3, material_index);
   return {};
}

} // namespace

std::expected<Scene, std::string> load_gltf(const std::filesystem::path& path) {
   Extras extras;
   auto parsed = parse_file(path, extras);
   const CameraLenses& lenses = extras.lenses;
   if (!parsed) {
      return std::unexpected(parsed.error());
   }
   const fastgltf::Asset& asset = *parsed;

   Scene scene;
   auto materials = load_materials(asset, scene);
   if (!materials) {
      return std::unexpected(materials.error());
   }
   scene.materials = std::move(*materials);
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
            const auto lens = lenses.find(*node.cameraIndex);
            auto camera = load_camera(
               asset.cameras[*node.cameraIndex],
               matrix,
               lens != lenses.end() ? lens->second : CameraLens{}
            );
            if (!camera) {
               error = camera.error();
               return;
            }
            scene.camera = *camera;
         }

         if (node.lightIndex.has_value()) {
            scene.punctual_lights.push_back(load_light(asset.lights[*node.lightIndex], matrix));
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

   if (!extras.environment.path.empty()) {
      auto environment = load_environment(
         path.parent_path() / extras.environment.path,
         extras.environment.strength,
         extras.environment.rotation_degrees * std::numbers::pi_v<float> / 180.0f
      );
      if (!environment) {
         return std::unexpected(environment.error());
      }
      scene.environment_image = std::move(*environment);
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
