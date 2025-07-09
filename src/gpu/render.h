#pragma once

#include "core/camera.h"
#include "core/vec3.h"

#include <vector>

std::vector<rt::Vec3> render(const rt::Camera& camera, int width, int height);
