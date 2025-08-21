#include "app/final_render.h"
#include "app/fly_camera.h"
#include "app/image_io.h"
#include "app/manifest.h"
#include "app/ui.h"
#include "app/window.h"
#include "core/aabb.h"
#include "core/camera.h"
#include "core/path_tracer.h"
#include "core/ray.h"
#include "core/scene_view.h"
#include "core/tonemap.h"
#include "core/vec3.h"
#include "gpu/renderer.h"
#include "scene/bvh_builder.h"
#include "scene/environment_map.h"
#include "scene/gltf_loader.h"
#include "scene/host_view.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <imgui.h>
#include <iostream>
#include <optional>
#include <ratio>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr float kMouseSensitivity = 0.003f; // radians par pixel
// Temps GPU de rendu à chaque image de l'interface
constexpr float kSliceMilliseconds = 30.0f;

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

std::string tone_mapping_name(rt::ToneMapping tone_mapping) {
   switch (tone_mapping) {
      case rt::ToneMapping::Agx:
         return "AgX (approximation de Benjamin Wrensch) puis sRGB";
      case rt::ToneMapping::Aces:
         return "ACES (approximation de Stephen Hill) puis sRGB";
      case rt::ToneMapping::Standard:
         break;
   }
   return "Standard (borné à [0, 1]) puis sRGB";
}

std::string caustic_description(const Renderer& renderer) {
   if (!renderer.caustics()) {
      return "path tracing seul";
   }
   const Renderer::CausticSettings settings = renderer.caustic_settings();
   return std::format(
      "photons progressifs : {} par passe, rayon initial {:.3g} pixels (au plus {:.4g}), alpha "
      "{:.3g}",
      settings.photons_per_pass,
      settings.initial_radius_pixels,
      settings.initial_radius,
      settings.alpha
   );
}

// PNG tone mappé, EXR linéaire et manifeste sous le même nom, plus l'image débruitée si le
// débruitage est actif. Renvoie le message à afficher.
std::string export_render(
   Renderer& renderer,
   const Renderer::DeviceInfo& device,
   const std::string& scene_path,
   Size size,
   const rt::CameraSettings& camera,
   const std::string& environment
) {
   const std::string
      base = std::format("render_{}x{}_{}spp", size.width, size.height, renderer.sample_count());
   const Renderer::AccumulatedStats accumulated = renderer.accumulated();
   const RenderManifest manifest{
      .date = current_local_time(),
      .device = device,
      .scene = scene_path,
      .width = size.width,
      .height = size.height,
      .samples_per_pixel = renderer.sample_count(),
      .gpu_milliseconds = accumulated.milliseconds,
      .rays = accumulated.rays,
      .mis = renderer.mis(),
      .firefly_clamp = renderer.firefly_clamp(),
      .camera = camera,
      .environment = environment,
      .environment_strength = renderer.environment_strength(),
      .environment_rotation = renderer.environment_rotation() * 180.0f / rt::kPi,
      .caustics = caustic_description(renderer),
      .denoiser = renderer.denoiser() ? "OptiX (albédo et normales)" : "aucun",
      .max_bounces = rt::kMaxBounces,
      .russian_roulette_start = rt::kRouletteStart,
      .bvh = "SAH (binning)",
      .tone_mapping = tone_mapping_name(renderer.display().tone_mapping),
      .exposure = renderer.display().exposure,
   };
   const std::vector<rt::Vec3> image = renderer.image();
   bool ok = write_png(base + ".png", image, size.width, size.height, renderer.display()) &&
             write_exr(base + ".exr", image, size.width, size.height) &&
             write_manifest(base + ".json", manifest);
   if (renderer.denoiser()) {
      const std::vector<rt::Vec3> denoised = renderer.denoised_image();
      ok = ok &&
           write_png(
              base + "_debruite.png",
              denoised,
              size.width,
              size.height,
              renderer.display()
           ) &&
           write_exr(base + "_debruite.exr", denoised, size.width, size.height);
   }
   return ok ? std::format("Exporté : {}", base) : std::format("Échec de l'export de {}", base);
}

// Mise au point sur l'objet au pixel (x, y) de l'image : distance du point touché le long de l'axe
// de visée
void focus_on_pixel(
   FlyCamera& fly_camera,
   const rt::Camera& camera,
   const rt::SceneView& scene,
   CursorPosition pixel
) {
   const rt::Ray ray = camera.generate_ray(
      static_cast<float>(pixel.x),
      static_cast<float>(pixel.y)
   );
   rt::Hit hit{};
   if (!rt::intersect_scene(ray, scene, hit)) {
      return;
   }
   const rt::CameraSettings settings = fly_camera.settings();
   const rt::Vec3 forward = rt::normalize(settings.target - settings.position);
   fly_camera.set_lens(settings.aperture_radius, hit.t * rt::dot(ray.direction, forward));
}

// Raccourcis selon la lettre imprimée sur la touche : ImGui traduit les touches GLFW selon la
// disposition du clavier
void handle_shortcuts(Window& window, Interface& ui, bool& export_requested) {
   const auto pressed = [](ImGuiKey key) { return ImGui::IsKeyPressed(key, false); };
   const FinalRender::State state = ui.final_render.state;
   if (pressed(ImGuiKey_P) && state != FinalRender::State::Running) {
      export_requested = true;
   }
   if (pressed(ImGuiKey_Escape) && state == FinalRender::State::Off) {
      window.close();
   }
}

} // namespace

int main(int argc, char** argv) {
   if (argc != 2 && argc != 4) {
      std::cerr << "Usage : " << argv[0] << " <scene.glb> [--hdri <environnement.exr|.hdr>]\n";
      return EXIT_FAILURE;
   }

   constexpr int kWidth = 800;
   constexpr int kHeight = 450;

   auto scene = rt::load_gltf(argv[1]);
   if (!scene) {
      std::cerr << "Failed to load GLTF file: " << scene.error() << '\n';
      return EXIT_FAILURE;
   }
   // HDRI donnée en ligne de commande : remplace celle de la scène
   if (argc == 4) {
      if (std::string_view(argv[2]) != "--hdri") {
         std::cerr << "Unknown option: " << argv[2] << '\n';
         return EXIT_FAILURE;
      }
      auto environment = rt::load_environment(argv[3], 1.0f, 0.0f);
      if (!environment) {
         std::cerr << environment.error() << '\n';
         return EXIT_FAILURE;
      }
      scene->environment_image = std::move(*environment);
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

   const rt::SceneView host_scene = rt::host_view(*scene, bvh);
   const rt::CameraSettings initial_camera = scene->camera.value_or(rt::CameraSettings{});
   FlyCamera fly_camera(initial_camera);
   rt::Camera camera(fly_camera.settings(), kWidth, kHeight);

   Interface ui;
   ui.scene_extent = scene_extent(*scene);
   ui.move_speed = 0.25f * ui.scene_extent;
   FinalRender& final_render = ui.final_render;
   double last_time = Window::time();
   CursorPosition last_cursor = window->cursor_position();

   // L'enregistrement CUDA doit être libéré avant qu'OpenGL ne réalloue le pixel buffer, puis
   // refait sur le nouveau
   const auto resize_image = [&](Size size) {
      renderer.detach_pixel_buffer();
      window->resize_image(size);
      renderer.resize(size.width, size.height);
      renderer.attach_pixel_buffer(window->pixel_buffer());
      camera = rt::Camera(fly_camera.settings(), size.width, size.height);
   };
   const auto export_image = [&] {
      ui.status = export_render(
         renderer,
         device,
         scene_path,
         window->image_size(),
         fly_camera.settings(),
         scene->environment_image.source
      );
      final_render.exported = final_render.state == FinalRender::State::Finished;
   };
   const auto advance_final_render = [&] {
      const bool sample_done = renderer.render_for(camera, kSliceMilliseconds);
      const std::uint32_t samples = renderer.sample_count();
      if (!sample_done || (!final_render.reached(samples) && !final_render.stop_requested)) {
         return;
      }
      final_render.finish(Window::time(), samples);
      renderer.refresh_display();
      if (final_render.export_when_finished) {
         export_image();
      }
   };

   while (!window->should_close()) {
      Window::poll_events();
      Window::begin_frame();

      // Fenêtre réduite : rien à afficher, mais le rendu final continue
      const Size size = window->framebuffer_size();
      if (size.width == 0 || size.height == 0) {
         ImGui::EndFrame();
         if (final_render.state == FinalRender::State::Running) {
            advance_final_render();
         } else {
            Window::wait_events();
         }
         continue;
      }
      if (!final_render.locked()) {
         const Size image_size = ui.resolution.size(size);
         if (image_size != window->image_size()) {
            resize_image(image_size);
         }
      }

      const double now = Window::time();
      const auto delta_time = static_cast<float>(now - last_time);
      last_time = now;

      bool camera_changed = draw_scene_panel(ui, renderer, fly_camera, initial_camera, size);
      const RenderPanelResult panel = draw_render_panel(ui, renderer, window->image_size());

      // Commandes inactives quand l'interface a le focus : saisie, copier-coller…
      const ImGuiIO& io = ImGui::GetIO();
      const bool interface_focused = io.WantCaptureKeyboard ||
                                     ImGui::IsWindowFocused(ImGuiFocusedFlags_AnyWindow);
      bool export_requested = panel.export_requested;
      if (!interface_focused) {
         handle_shortcuts(*window, ui, export_requested);
      }
      switch (panel.final_render) {
         case FinalRender::Action::Start:
            renderer.reset();
            final_render.start(now, 0);
            ui.status.clear();
            break;
         case FinalRender::Action::Continue:
            final_render.start(now, renderer.sample_count());
            ui.status.clear();
            break;
         case FinalRender::Action::Stop:
            final_render.stop_requested = true;
            break;
         case FinalRender::Action::Cancel:
            renderer.reset();
            final_render.state = FinalRender::State::Off;
            break;
         case FinalRender::Action::Close:
            final_render.state = FinalRender::State::Off;
            break;
         case FinalRender::Action::Export:
            export_requested = true;
            break;
         case FinalRender::Action::None:
            break;
      }
      if (export_requested) {
         export_image();
      }

      if (final_render.state == FinalRender::State::Running) {
         advance_final_render();
      } else if (final_render.state == FinalRender::State::Finished) {
         renderer.refresh_display();
      } else {
         // Un clic sur l'image rend le clavier à la caméra (ImGui ne le fait que pour le clic
         // gauche)
         if (!io.WantCaptureMouse && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
            ImGui::SetWindowFocus(nullptr);
         }
         if (!io.WantCaptureMouse && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            if (const std::optional<CursorPosition> pixel = window->cursor_in_image()) {
               focus_on_pixel(fly_camera, camera, host_scene, *pixel);
               camera_changed = true;
            }
         }
         const rt::Vec3 direction = interface_focused ? rt::Vec3{0.0f, 0.0f, 0.0f}
                                                      : movement_direction(*window);
         if (direction != rt::Vec3{0.0f, 0.0f, 0.0f}) {
            const float speed = ui.move_speed * (window->is_down(Key::Fast) ? 4.0f : 1.0f);
            fly_camera.move(rt::normalize(direction) * (speed * delta_time));
            camera_changed = true;
         }
         const CursorPosition cursor = window->cursor_position();
         if (window->is_looking() && (cursor.x != last_cursor.x || cursor.y != last_cursor.y)) {
            fly_camera.rotate(
               static_cast<float>(cursor.x - last_cursor.x) * kMouseSensitivity,
               static_cast<float>(last_cursor.y - cursor.y) * kMouseSensitivity
            );
            camera_changed = true;
         }
         if (camera_changed) {
            const Size image_size = window->image_size();
            camera = rt::Camera(fly_camera.settings(), image_size.width, image_size.height);
            renderer.reset();
         }
         renderer.render_for(camera, kSliceMilliseconds);
      }
      last_cursor = window->cursor_position();
      window->present();
   }

   return EXIT_SUCCESS;
}
