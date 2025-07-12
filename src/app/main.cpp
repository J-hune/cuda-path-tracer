#include "app/image_io.h"
#include "core/camera.h"
#include "core/vec3.h"
#include "gpu/render.h"
#include "scene/bvh_builder.h"
#include "scene/gltf_loader.h"

#include <cstdlib>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
   if (argc < 2) {
      std::cerr << "Usage : " << argv[0] << " <scene.glb>\n";
      return EXIT_FAILURE;
   }

   constexpr int kWidth = 800;
   constexpr int kHeight = 450;

   const auto scene = rt::load_gltf(argv[1]);
   if (!scene) {
      std::cerr << "Failed to load GLTF file: " << scene.error() << '\n';
      return EXIT_FAILURE;
   }

   // On affiche le nombre de sommets et de triangles dans la scène
   std::cout << "Loaded GLTF file: " << argv[1] << '\n';
   std::cout << "Number of vertices: " << scene->positions.size() << '\n';
   std::cout << "Number of triangles: " << scene->indices.size() / 3 << '\n';

   const rt::Camera camera(scene->camera.value_or(rt::CameraSettings{}), kWidth, kHeight);

   const rt::Bvh bvh = rt::build_bvh(*scene);
   const std::vector<rt::Vec3> image = render(camera, *scene, bvh, kWidth, kHeight);
   if (!write_png("image.png", image, kWidth, kHeight)) {
      std::cerr << "Failed to write image.png" << '\n';
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
