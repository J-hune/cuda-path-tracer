#include "app/fly_camera.h"
#include "app/image_io.h"
#include "app/manifest.h"
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
#include <cstdlib>
#include <format>
#include <imgui.h>
#include <iostream>
#include <ratio>
#include <string>
#include <string_view>
#include <utility>
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

// État de l'interface qui n'appartient pas au renderer
struct Controls {
   float firefly_threshold = rt::kDefaultFireflyClamp;
   bool export_requested = false;
   std::string status; // résultat du dernier export
};

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
   const rt::CameraSettings& camera
) {
   const std::string base = std::format("render_{}spp", renderer.sample_count());
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
      .aperture_radius = camera.aperture_radius,
      .focus_distance = camera.focus_distance,
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

// Réglages de la lentille. Renvoie vrai s'ils ont changé.
bool draw_lens_settings(FlyCamera& fly_camera) {
   const rt::CameraSettings settings = fly_camera.settings();
   float aperture = settings.aperture_radius;
   float focus = settings.focus_distance;
   ImGui::SeparatorText("Caméra");
   bool changed = ImGui::SliderFloat(
      "Aperture",
      &aperture,
      0.0f,
      1.0f,
      "%.3f",
      ImGuiSliderFlags_Logarithmic
   );
   changed |= ImGui::SliderFloat(
      "Focus",
      &focus,
      0.1f,
      100.0f,
      "%.2f",
      ImGuiSliderFlags_Logarithmic
   );
   ImGui::TextUnformatted("Clic gauche : focus sur l'objet");
   if (changed) {
      fly_camera.set_lens(aperture, focus);
   }
   return changed;
}

// Mise au point sur l'objet sous le curseur : distance du point touché le long de l'axe de visée
void focus_on_cursor(
   FlyCamera& fly_camera,
   const rt::Camera& camera,
   const rt::SceneView& scene,
   CursorPosition cursor
) {
   const ImVec2 scale = ImGui::GetIO().DisplayFramebufferScale;
   const rt::Ray ray = camera.generate_ray(
      static_cast<float>(cursor.x) * scale.x,
      static_cast<float>(cursor.y) * scale.y
   );
   rt::Hit hit{};
   if (!rt::intersect_scene(ray, scene, hit)) {
      return;
   }
   const rt::CameraSettings settings = fly_camera.settings();
   const rt::Vec3 forward = rt::normalize(settings.target - settings.position);
   fly_camera.set_lens(settings.aperture_radius, hit.t * rt::dot(ray.direction, forward));
}

// Panneau de l'interface. Renvoie vrai si les réglages de la lentille ont changé.
bool draw_panel(Renderer& renderer, Controls& controls, FlyCamera& fly_camera, Size image_size) {
   ImGui::SetNextWindowPos({10.0f, 10.0f}, ImGuiCond_FirstUseEver);
   ImGui::Begin("Rendu", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

   const ImGuiIO& io = ImGui::GetIO();
   const Renderer::AccumulatedStats accumulated = renderer.accumulated();
   const double mrays_per_second = accumulated.milliseconds > 0.0
                                      ? static_cast<double>(accumulated.rays) /
                                           (accumulated.milliseconds * 1e3)
                                      : 0.0;
   ImGui::Text("%.0f FPS (%.1f ms)", io.Framerate, 1000.0f / io.Framerate);
   ImGui::Text("%d × %d pixels", image_size.width, image_size.height);
   ImGui::Text("%u spp", renderer.sample_count());
   ImGui::Text("%.0f Mrays/s", mrays_per_second);

   ImGui::SeparatorText("Réglages");
   bool mis = renderer.mis();
   if (ImGui::Checkbox("MIS (M)", &mis)) {
      renderer.set_mis(mis);
   }
   bool clamp = renderer.firefly_clamp() > 0.0f;
   if (ImGui::Checkbox("Firefly clamp", &clamp)) {
      renderer.set_firefly_clamp(clamp ? controls.firefly_threshold : 0.0f);
   }
   if (clamp && ImGui::SliderFloat(
                   "Seuil",
                   &controls.firefly_threshold,
                   1.0f,
                   100.0f,
                   "%.1f",
                   ImGuiSliderFlags_Logarithmic
                )) {
      renderer.set_firefly_clamp(controls.firefly_threshold);
   }
   bool caustics = renderer.caustics();
   if (ImGui::Checkbox("Caustiques (photons)", &caustics)) {
      renderer.set_caustics(caustics);
   }
   bool denoise = renderer.denoiser();
   if (ImGui::Checkbox("Denoiser OptiX", &denoise)) {
      renderer.set_denoiser(denoise);
   }
   int layer = static_cast<int>(renderer.display_layer());
   if (ImGui::Combo("AOV", &layer, "Beauty\0Albedo\0Normal\0")) {
      renderer.set_display_layer(static_cast<Renderer::DisplayLayer>(layer));
   }

   const bool lens_changed = draw_lens_settings(fly_camera);

   if (renderer.has_environment_map()) {
      ImGui::SeparatorText("Environnement");
      float strength = renderer.environment_strength();
      float rotation_degrees = renderer.environment_rotation() * 180.0f / rt::kPi;
      bool changed = ImGui::SliderFloat(
         "Intensité",
         &strength,
         0.0f,
         10.0f,
         "%.2f",
         ImGuiSliderFlags_Logarithmic
      );
      changed |= ImGui::SliderFloat("Rotation", &rotation_degrees, -180.0f, 180.0f, "%.0f°");
      if (changed) {
         renderer.set_environment(strength, rotation_degrees * rt::kPi / 180.0f);
      }
   }

   ImGui::SeparatorText("Affichage");
   rt::DisplaySettings display = renderer.display();
   int tone_mapping = static_cast<int>(display.tone_mapping);
   bool display_changed = ImGui::Combo("Tone mapping", &tone_mapping, "AgX\0ACES\0Standard\0");
   display_changed |= ImGui::SliderFloat(
      "Exposition (EV)",
      &display.exposure,
      -10.0f,
      10.0f,
      "%.1f"
   );
   if (display_changed) {
      display.tone_mapping = static_cast<rt::ToneMapping>(tone_mapping);
      renderer.set_display(display);
   }

   ImGui::SeparatorText("Export");
   if (ImGui::Button("Exporter (P)")) {
      controls.export_requested = true;
   }
   if (!controls.status.empty()) {
      ImGui::TextUnformatted(controls.status.c_str());
   }

   ImGui::SeparatorText("Commandes");
   ImGui::TextUnformatted(
      "ZQSD : se déplacer\n"
      "Espace, Ctrl : monter, descendre\n"
      "Maj : plus vite\n"
      "Clic droit : regarder\n"
      "Échap : quitter"
   );
   ImGui::End();
   return lens_changed;
}

// Raccourcis selon la lettre imprimée sur la touche : ImGui traduit les touches GLFW selon la
// disposition du clavier
void handle_shortcuts(Window& window, Renderer& renderer, Controls& controls) {
   if (ImGui::GetIO().WantCaptureKeyboard) {
      return;
   }
   const auto pressed = [](ImGuiKey key) { return ImGui::IsKeyPressed(key, false); };
   if (pressed(ImGuiKey_Escape)) {
      window.close();
   }
   if (pressed(ImGuiKey_P)) {
      controls.export_requested = true;
   }
   if (pressed(ImGuiKey_M)) {
      renderer.set_mis(!renderer.mis());
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
   FlyCamera fly_camera(scene->camera.value_or(rt::CameraSettings{}));
   rt::Camera camera(fly_camera.settings(), kWidth, kHeight);
   // Vitesse de déplacement proportionnelle à la taille de la scène
   const float move_speed = 0.25f * scene_extent(*scene);

   double last_time = Window::time();
   Controls controls;
   CursorPosition last_cursor = window->cursor_position();
   while (!window->should_close()) {
      Window::poll_events();
      Window::begin_frame();

      // Redimensionnement : l'enregistrement CUDA doit être libéré avant qu'OpenGL ne réalloue
      // le pixel buffer, puis refait sur le nouveau
      const Size size = window->framebuffer_size();
      if (size.width == 0 || size.height == 0) {
         ImGui::EndFrame();
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

      bool lens_changed = draw_panel(renderer, controls, fly_camera, window->image_size());
      handle_shortcuts(*window, renderer, controls);
      if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGui::GetIO().WantCaptureMouse) {
         focus_on_cursor(fly_camera, camera, host_scene, window->cursor_position());
         lens_changed = true;
      }
      if (controls.export_requested) {
         controls.status = export_render(
            renderer,
            device,
            scene_path,
            window->image_size(),
            fly_camera.settings()
         );
         controls.export_requested = false;
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

      if (moved || lens_changed) {
         camera = rt::Camera(fly_camera.settings(), size.width, size.height);
         renderer.reset();
      }

      renderer.render(camera);
      window->present();
   }

   return EXIT_SUCCESS;
}
