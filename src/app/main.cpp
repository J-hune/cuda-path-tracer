#include "app/fly_camera.h"
#include "app/image_io.h"
#include "app/window.h"
#include "core/aabb.h"
#include "core/camera.h"
#include "core/vec3.h"
#include "gpu/renderer.h"
#include "scene/bvh_builder.h"
#include "scene/gltf_loader.h"

#include <cstdlib>
#include <format>
#include <iostream>
#include <string>

namespace {

constexpr float kMouseSensitivity = 0.003f; // radians par pixel

// Longueur de la diagonale de la boîte englobant la scène
float scene_extent(const rt::Scene& scene) {
   rt::Aabb bounds;
   for (const rt::Vec3& position : scene.positions) {
      bounds.grow(position);
   }
   return scene.positions.empty() ? 1.0f : rt::length(bounds.max - bounds.min);
}

// Direction de déplacement demandée au clavier, dans le repère de la caméra
rt::Vec3 movement_direction(const Window& window) {
   rt::Vec3 direction{0.0f, 0.0f, 0.0f};
   direction.z += window.is_down(Key::Forward) ? 1.0f : 0.0f;
   direction.z -= window.is_down(Key::Backward) ? 1.0f : 0.0f;
   direction.x += window.is_down(Key::Right) ? 1.0f : 0.0f;
   direction.x -= window.is_down(Key::Left) ? 1.0f : 0.0f;
   direction.y += window.is_down(Key::Up) ? 1.0f : 0.0f;
   direction.y -= window.is_down(Key::Down) ? 1.0f : 0.0f;
   return direction;
}

} // namespace

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

   const rt::Bvh bvh = rt::build_bvh(*scene);

   // La fenêtre (et son contexte OpenGL) doit exister avant le renderer, et lui survivre
   auto window = Window::create(kWidth, kHeight, "raytracer");
   if (!window) {
      std::cerr << window.error() << '\n';
      return EXIT_FAILURE;
   }
   Renderer renderer(*scene, bvh, kWidth, kHeight);
   renderer.attach_pixel_buffer(window->pixel_buffer());

   FlyCamera fly_camera(scene->camera.value_or(rt::CameraSettings{}));
   rt::Camera camera(fly_camera.settings(), kWidth, kHeight);
   // Vitesse de déplacement proportionnelle à la taille de la scène
   const float move_speed = 0.25f * scene_extent(*scene);

   double last_time = Window::time();
   CursorPosition last_cursor = window->cursor_position();
   while (!window->should_close()) {
      Window::poll_events();
      const double now = Window::time();
      const auto delta_time = static_cast<float>(now - last_time);
      last_time = now;

      if (window->was_pressed(Key::Quit)) {
         window->close();
      }
      if (window->was_pressed(Key::Export)) {
         const std::string path = std::format("render_{}spp.png", renderer.sample_count());
         if (write_png(path, renderer.image(), kWidth, kHeight)) {
            std::cout << "Image exportée : " << path << '\n';
         } else {
            std::cerr << "Échec de l'export de " << path << '\n';
         }
      }

      bool moved = false;
      const rt::Vec3 direction = movement_direction(*window);
      if (direction != rt::Vec3{0.0f, 0.0f, 0.0f}) {
         const float speed = move_speed * (window->is_down(Key::Fast) ? 4.0f : 1.0f);
         fly_camera.move(rt::normalize(direction) * (speed * delta_time));
         moved = true;
      }

      const CursorPosition cursor = window->cursor_position();
      if (window->is_looking() && (cursor.x != last_cursor.x || cursor.y != last_cursor.y)) {
         fly_camera.rotate(
            static_cast<float>(cursor.x - last_cursor.x) * kMouseSensitivity,
            static_cast<float>(last_cursor.y - cursor.y) * kMouseSensitivity
         );
         moved = true;
      }
      last_cursor = cursor;

      if (moved) {
         camera = rt::Camera(fly_camera.settings(), kWidth, kHeight);
         renderer.reset();
      }

      renderer.render(camera);
      window->present();
      window->set_title(std::format("raytracer - {} spp", renderer.sample_count()));
   }

   return EXIT_SUCCESS;
}
