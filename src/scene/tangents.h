#pragma once

#include "scene/gltf_loader.h"

#include <cstddef>
#include <cstdint>

namespace rt {

// Tangentes des sommets [first_vertex, first_vertex + vertex_count) à partir des positions et des
// coordonnées de texture de leurs triangles (les index_count derniers indices de la scène),
// ajoutées à scene.tangents. glTF place l'origine des textures en haut à gauche et le vert des
// textures de normales vers le haut : la bitangente suit les v décroissants.
void compute_tangents(
   Scene& scene,
   std::uint32_t first_vertex,
   std::uint32_t vertex_count,
   std::size_t index_count
);

} // namespace rt
