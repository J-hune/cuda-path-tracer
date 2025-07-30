#include "app/fly_camera.h"
#include "app/image_io.h"
#include "app/manifest.h"
#include "app/window.h"
#include "core/aabb.h"
#include "core/camera.h"
#include "core/path_tracer.h"
#include "core/tonemap.h"
#include "core/vec3.h"
#include "gpu/renderer.h"
#include "scene/bvh_builder.h"
#include "scene/gltf_loader.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <ratio>
#include <string>
#include <vector>

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

   const auto build_start = std::chrono::steady_clock::now();
   const rt::Bvh bvh = rt::build_bvh(*scene);
   const std::chrono::duration<double, std::milli> build_time = std::chrono::steady_clock::now() -
                                                                build_start;
   const rt::BvhStats stats = rt::compute_stats(bvh);
   std::cout << std::format(
      "BVH (SAH) : {:.0f} ms, {} nœuds, {} feuilles de {:.2f} triangles, profondeur moyenne {:.1f} "
      "(max {})\n",
      build_time.count(),
      stats.node_count,
      stats.leaf_count,
      stats.average_leaf_size,
      stats.average_depth,
      stats.max_depth
   );

   // La fenêtre (et son contexte OpenGL) doit exister avant le renderer, et lui survivre
   auto window = Window::create(kWidth, kHeight, "raytracer");
   if (!window) {
      std::cerr << window.error() << '\n';
      return EXIT_FAILURE;
   }
   Renderer renderer(*scene, bvh, kWidth, kHeight);
   const Renderer::DeviceInfo device = Renderer::device_info();
   const std::string scene_path = argv[1];
   renderer.attach_pixel_buffer(window->pixel_buffer());

   FlyCamera fly_camera(scene->camera.value_or(rt::CameraSettings{}));
   rt::Camera camera(fly_camera.settings(), kWidth, kHeight);
   // Vitesse de déplacement proportionnelle à la taille de la scène
   const float move_speed = 0.25f * scene_extent(*scene);

   double last_time = Window::time();
   double stats_start = last_time;
   std::uint64_t stats_rays = 0;
   double stats_milliseconds = 0.0;
   CursorPosition last_cursor = window->cursor_position();
   while (!window->should_close()) {
      Window::poll_events();

      // Redimensionnement : l'enregistrement CUDA doit être libéré avant qu'OpenGL ne réalloue
      // le pixel buffer, puis refait sur le nouveau
      const Size size = window->framebuffer_size();
      if (size.width == 0 || size.height == 0) {
         Window::wait_events();
         continue;
      }
      if (size != window->image_size()) {
         renderer.detach_pixel_buffer();
         window->resize_image(size);
         renderer.resize(size.width, size.height);
         renderer.attach_pixel_buffer(window->pixel_buffer());
         camera = rt::Camera(fly_camera.settings(), size.width, size.height);
      }

      const double now = Window::time();
      const auto delta_time = static_cast<float>(now - last_time);
      last_time = now;

      if (window->was_pressed(Key::Quit)) {
         window->close();
      }
      if (window->was_pressed(Key::Export)) {
         // PNG tone mappé (ce qu'affiche la fenêtre), EXR linéaire et manifeste, sous le même nom
         const std::string base = std::format("render_{}spp", renderer.sample_count());
         const Size image_size = window->image_size();
         const std::vector<rt::Vec3> image = renderer.image();
         const Renderer::AccumulatedStats accumulated = renderer.accumulated();
         const RenderManifest manifest{
            .date = current_local_time(),
            .device = device,
            .scene = scene_path,
            .width = image_size.width,
            .height = image_size.height,
            .samples_per_pixel = renderer.sample_count(),
            .gpu_milliseconds = accumulated.milliseconds,
            .rays = accumulated.rays,
            .mis = renderer.mis(),
            .firefly_clamp = renderer.firefly_clamp(),
            .max_bounces = rt::kMaxBounces,
            .russian_roulette_start = rt::kRouletteStart,
            .bvh = "SAH (binning)",
            .tone_mapping = "ACES (approximation de Stephen Hill) puis sRGB",
            .exposure = rt::kDefaultExposure,
         };
         const bool png = write_png(base + ".png", image, image_size.width, image_size.height);
         const bool exr = write_exr(base + ".exr", image, image_size.width, image_size.height);
         const bool json = write_manifest(base + ".json", manifest);
         if (png && exr && json) {
            std::cout << "Image exportée : " << base << ".png, .exr et .json\n";
         } else {
            std::cerr << "Échec de l'export de " << base << '\n';
         }
      }

      if (window->was_pressed(Key::ToggleMis)) {
         renderer.set_mis(!renderer.mis());
      }
      if (window->was_pressed(Key::ToggleFireflyClamp)) {
         renderer.set_firefly_clamp(
            renderer.firefly_clamp() > 0.0f ? 0.0f : rt::kDefaultFireflyClamp
         );
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
         camera = rt::Camera(fly_camera.settings(), size.width, size.height);
         renderer.reset();
      }

      renderer.render(camera);
      window->present();
      // Débit moyenné sur une demi-seconde, pour un affichage lisible
      const Renderer::FrameStats frame = renderer.last_frame();
      stats_rays += frame.rays;
      stats_milliseconds += frame.milliseconds;
      if (now - stats_start >= 0.5) {
         const double mrays_per_second = static_cast<double>(stats_rays) /
                                         (stats_milliseconds * 1e3);
         window->set_title(
            std::format(
               "raytracer - {} spp - {:.0f} Mrayons/s",
               renderer.sample_count(),
               mrays_per_second
            )
         );
         stats_start = now;
         stats_rays = 0;
         stats_milliseconds = 0.0;
      }
   }

   return EXIT_SUCCESS;
}
