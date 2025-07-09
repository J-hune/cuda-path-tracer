#include "app/image_io.h"
#include "core/camera.h"
#include "core/vec3.h"
#include "gpu/render.h"

#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
   constexpr int kWidth = 800;
   constexpr int kHeight = 450;
   const rt::Camera camera(rt::CameraSettings{}, kWidth, kHeight);

   const std::vector<rt::Vec3> image = render(camera, kWidth, kHeight);
   if (!write_png("image.png", image, kWidth, kHeight)) {
      std::cerr << "Failed to write image.png" << '\n';
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
