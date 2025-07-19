#include "app/window.h"
#include "core/camera.h"
#include "gpu/renderer.h"
#include "scene/bvh_builder.h"
#include "scene/gltf_loader.h"

#include <cstdlib>
#include <format>
#include <iostream>

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

   // La fenêtre (et son contexte OpenGL) doit exister avant le renderer, et lui survivre
   auto window = Window::create(kWidth, kHeight, "raytracer");
   if (!window) {
      std::cerr << window.error() << '\n';
      return EXIT_FAILURE;
   }
   Renderer renderer(*scene, bvh, kWidth, kHeight);
   renderer.attach_pixel_buffer(window->pixel_buffer());

   while (!window->should_close()) {
      renderer.render(camera);
      window->present();
      window->set_title(std::format("raytracer - {} spp", renderer.sample_count()));
      Window::poll_events();
   }

   return EXIT_SUCCESS;
}
