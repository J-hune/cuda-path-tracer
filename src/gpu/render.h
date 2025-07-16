#pragma once

#include "core/camera.h"
#include "core/vec3.h"
#include "scene/bvh_builder.h"
#include "scene/gltf_loader.h"

#include <vector>

std::vector<rt::Vec3> render(
   const rt::Camera& camera,
   const rt::Scene& scene,
   const rt::Bvh& bvh,
   int width,
   int height
);
