#pragma once

#include "scene/gltf_loader.h"

#include <cstdint>
#include <vector>

namespace rt {

// Répartition des lumières selon leur puissance, pour pick_light : fonction de répartition sur les
// triangles émissifs puis les lumières ponctuelles, et index de lumière de chaque triangle
struct LightDistribution {
   std::vector<float> cdf;                   // nombre de lumières + 1 entrées, de 0 à 1
   std::vector<std::int32_t> triangle_light; // -1 si le triangle n'est pas une lumière
};

LightDistribution light_distribution(const Scene& scene);

} // namespace rt
