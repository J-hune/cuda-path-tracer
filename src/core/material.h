#pragma once

#include "core/vec3.h"

#include <cstdint>

namespace rt {

struct Material {
   Vec3 albedo;
   Vec3 emission;
   // Index dans Scene::textures, -1 si aucune. Metallic-roughness : rugosité dans le canal vert,
   // métallique dans le bleu, multipliés par roughness et metallic. Émission : multipliée par
   // emission.
   std::int32_t base_color_texture = -1;
   std::int32_t metallic_roughness_texture = -1;
   std::int32_t emissive_texture = -1;
   // Texture de normales (espace tangent) et son intensité (scale de glTF)
   std::int32_t normal_texture = -1;
   float normal_scale = 1.0f;
   // Modèle metallic-roughness de glTF (diélectrique mat par défaut)
   float metallic = 0.0f;
   float roughness = 1.0f;
   // 0 : opaque (diffus), 1 : diélectrique transparent (verre)
   float transmission = 0.0f;
   // Indice de réfraction
   float ior = 1.5f;
   // Coefficient d'absorption du volume par unité de distance (Beer-Lambert)
   Vec3 absorption{0.0f, 0.0f, 0.0f};
   // 20 / nombre d'Abbe, 0 : indice indépendant de la longueur d'onde
   float dispersion = 0.0f;
};

} // namespace rt
