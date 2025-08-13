#pragma once

#include "core/scene_view.h"
#include "scene/bvh_builder.h"
#include "scene/gltf_loader.h"

#include <cstdint>

namespace rt {

// Vue de la scène sur les tableaux du CPU, pour y lancer quelques rayons sans passer par le GPU
inline SceneView host_view(const Scene& scene, const Bvh& bvh) {
   SceneView view;
   view.nodes = bvh.nodes.empty() ? nullptr : bvh.nodes.data();
   view.triangle_indices = bvh.triangle_indices.data();
   view.bvh_triangles = bvh.triangles.data();
   view.positions = scene.positions.data();
   view.indices = scene.indices.data();
   view.materials = scene.materials.data();
   view.material_indices = scene.material_indices.data();
   view.emissive_triangles = scene.emissive_triangles.data();
   view.emissive_triangle_count = static_cast<std::uint32_t>(scene.emissive_triangles.size());
   view.punctual_lights = scene.punctual_lights.data();
   view.punctual_light_count = static_cast<std::uint32_t>(scene.punctual_lights.size());
   view.uvs = scene.uvs.data();
   view.normals = scene.normals.empty() ? nullptr : scene.normals.data();
   view.textures = scene.textures.data();
   view.texels = scene.texels.data();
   view.environment = scene.environment;
   return view;
}

} // namespace rt
