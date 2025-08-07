#pragma once

#include "core/photon_map.h"
#include "scene/gltf_loader.h"

#include <vector>

namespace rt {

// Une sphère par matériau qui crée des caustiques (verre, métal poli), englobant ses triangles
std::vector<PhotonTarget> photon_targets(const Scene& scene);

} // namespace rt
