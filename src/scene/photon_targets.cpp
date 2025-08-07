#include "scene/photon_targets.h"

#include "core/aabb.h"
#include "core/material.h"
#include "core/path_tracer.h"
#include "core/photon_map.h"
#include "core/vec3.h"
#include "scene/gltf_loader.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rt {

std::vector<PhotonTarget> photon_targets(const Scene& scene) {
   std::vector<Aabb> bounds(scene.materials.size());
   std::vector<bool> used(scene.materials.size(), false);
   for (std::size_t triangle = 0; triangle < scene.material_indices.size(); ++triangle) {
      const std::uint32_t m = scene.material_indices[triangle];
      const Material& material = scene.materials[m];
      if (material.transmission <= 0.0f && !is_glossy_metal(material)) {
         continue;
      }
      used[m] = true;
      for (std::size_t corner = 0; corner < 3; ++corner) {
         bounds[m].grow(scene.positions[scene.indices[(3 * triangle) + corner]]);
      }
   }

   std::vector<PhotonTarget> targets;
   for (std::size_t m = 0; m < bounds.size(); ++m) {
      if (used[m]) {
         targets.push_back({
            .center = 0.5f * (bounds[m].min + bounds[m].max),
            .radius = 0.5f * length(bounds[m].max - bounds[m].min),
         });
      }
   }
   return targets;
}

} // namespace rt
