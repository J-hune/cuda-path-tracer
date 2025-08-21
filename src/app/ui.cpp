#include "app/ui.h"

#include "app/final_render.h"
#include "app/fly_camera.h"
#include "app/window.h"
#include "core/camera.h"
#include "core/tonemap.h"
#include "core/vec3.h"
#include "gpu/renderer.h"

#include <algorithm>
#include <array>
#include <format>
#include <imgui.h>
#include <string>
#include <utility>

namespace {

struct Preset {
   const char* name = nullptr;
   Size size; // 0 × 0 : taille de la fenêtre ou taille personnalisée
};

constexpr std::array<Preset, 6> kPresets{{
   {.name = "Fenêtre", .size = {}},
   {.name = "720p", .size = {1280, 720}},
   {.name = "1080p", .size = {1920, 1080}},
   {.name = "1440p", .size = {2560, 1440}},
   {.name = "4K", .size = {3840, 2160}},
   {.name = "Personnalisée", .size = {}},
}};
constexpr int kWindowPreset = 0;
constexpr int kCustomPreset = 5;
constexpr int kMinSide = 16;
constexpr int kMaxSide = 4096; // au-delà, les tampons et le débruiteur dépassent la mémoire du GPU
constexpr float kRadiansToDegrees = 180.0f / rt::kPi;

std::string preset_label(int index, Size window) {
   if (index == kCustomPreset) {
      return kPresets[kCustomPreset].name;
   }
   const Size size = index == kWindowPreset ? window : kPresets[index].size;
   return std::format("{} × {} ({})", size.width, size.height, kPresets[index].name);
}

void draw_resolution(Resolution& resolution, Size window) {
   ImGui::SeparatorText("Résolution");
   if (ImGui::BeginCombo("Taille", preset_label(resolution.preset, window).c_str())) {
      for (int i = 0; std::cmp_less(i, kPresets.size()); ++i) {
         const bool selected = i == resolution.preset;
         if (ImGui::Selectable(preset_label(i, window).c_str(), selected)) {
            resolution.preset = i;
         }
         if (selected) {
            ImGui::SetItemDefaultFocus();
         }
      }
      ImGui::EndCombo();
   }
   if (resolution.preset == kCustomPreset) {
      std::array<int, 2> size{resolution.custom.width, resolution.custom.height};
      if (ImGui::InputInt2("Largeur, hauteur", size.data())) {
         resolution.custom = {
            .width = std::clamp(size[0], kMinSide, kMaxSide),
            .height = std::clamp(size[1], kMinSide, kMaxSide),
         };
      }
   }
}

// Réglages de la caméra. Renvoie vrai s'ils ont changé.
bool draw_camera(Interface& ui, FlyCamera& camera, const rt::CameraSettings& initial) {
   ImGui::SeparatorText("Caméra");
   bool changed = false;

   const rt::Vec3 position = camera.position();
   std::array<float, 3> xyz{position.x, position.y, position.z};
   if (ImGui::DragFloat3("Position", xyz.data(), 0.002f * ui.scene_extent, 0.0f, 0.0f, "%.3f")) {
      camera.set_position({xyz[0], xyz[1], xyz[2]});
      changed = true;
   }

   float yaw = camera.yaw() * kRadiansToDegrees;
   float pitch = camera.pitch() * kRadiansToDegrees;
   const float max_pitch = FlyCamera::max_pitch() * kRadiansToDegrees;
   bool turned = ImGui::DragFloat(
      "Yaw",
      &yaw,
      0.2f,
      -180.0f,
      180.0f,
      "%.1f°",
      ImGuiSliderFlags_WrapAround
   );
   ImGui::SetItemTooltip("Rotation autour de la verticale, 0° : vers -z");
   turned |= ImGui::DragFloat(
      "Pitch",
      &pitch,
      0.2f,
      -max_pitch,
      max_pitch,
      "%.1f°",
      ImGuiSliderFlags_AlwaysClamp
   );
   ImGui::SetItemTooltip("Inclinaison vers le haut (positive) ou le bas");
   if (turned) {
      camera.set_orientation(yaw / kRadiansToDegrees, pitch / kRadiansToDegrees);
      changed = true;
   }

   const rt::CameraSettings settings = camera.settings();
   float fov = settings.vertical_fov_degrees;
   if (ImGui::SliderFloat("FOV", &fov, 5.0f, 120.0f, "%.1f°")) {
      camera.set_vertical_fov(fov);
      changed = true;
   }
   ImGui::SetItemTooltip("Champ de vision vertical");
   float aperture = settings.aperture_radius;
   float focus = settings.focus_distance;
   bool lens_changed = ImGui::SliderFloat(
      "Aperture",
      &aperture,
      0.0f,
      1.0f,
      "%.3f",
      ImGuiSliderFlags_Logarithmic
   );
   lens_changed |= ImGui::SliderFloat(
      "Focus",
      &focus,
      0.1f,
      100.0f,
      "%.2f",
      ImGuiSliderFlags_Logarithmic
   );
   ImGui::SetItemTooltip("Clic gauche sur l'image : focus sur l'objet");
   if (lens_changed) {
      camera.set_lens(aperture, focus);
      changed = true;
   }

   ImGui::SliderFloat(
      "Vitesse",
      &ui.move_speed,
      0.01f * ui.scene_extent,
      4.0f * ui.scene_extent,
      "%.3g par seconde",
      ImGuiSliderFlags_Logarithmic
   );
   ImGui::SetItemTooltip("Vitesse de déplacement au clavier, quadruplée avec Maj");
   if (ImGui::Button("Reset")) {
      camera = FlyCamera(initial);
      changed = true;
   }
   ImGui::SetItemTooltip("Revient à la caméra de la scène");
   return changed;
}

void draw_sampling(Interface& ui, Renderer& renderer) {
   ImGui::SeparatorText("Sampling");
   bool mis = renderer.mis();
   if (ImGui::Checkbox("MIS", &mis)) {
      renderer.set_mis(mis);
   }
   bool clamp = renderer.firefly_clamp() > 0.0f;
   if (ImGui::Checkbox("Firefly clamp", &clamp)) {
      renderer.set_firefly_clamp(clamp ? ui.firefly_threshold : 0.0f);
   }
   if (clamp && ImGui::SliderFloat(
                   "Seuil",
                   &ui.firefly_threshold,
                   1.0f,
                   100.0f,
                   "%.1f",
                   ImGuiSliderFlags_Logarithmic
                )) {
      renderer.set_firefly_clamp(ui.firefly_threshold);
   }
   bool caustics = renderer.caustics();
   if (ImGui::Checkbox("Caustiques (photons)", &caustics)) {
      renderer.set_caustics(caustics);
   }
}

void draw_environment(Renderer& renderer) {
   if (!renderer.has_environment_map()) {
      return;
   }
   ImGui::SeparatorText("Environnement");
   float strength = renderer.environment_strength();
   float rotation_degrees = renderer.environment_rotation() * kRadiansToDegrees;
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
      renderer.set_environment(strength, rotation_degrees / kRadiansToDegrees);
   }
}

void draw_statistics(const Renderer& renderer, Size image) {
   const ImGuiIO& io = ImGui::GetIO();
   const Renderer::AccumulatedStats accumulated = renderer.accumulated();
   const double mrays_per_second = accumulated.milliseconds > 0.0
                                      ? static_cast<double>(accumulated.rays) /
                                           (accumulated.milliseconds * 1e3)
                                      : 0.0;
   ImGui::Text("%.0f FPS (%.1f ms)", io.Framerate, 1000.0f / io.Framerate);
   ImGui::Text("%d × %d pixels", image.width, image.height);
   ImGui::Text("%u spp", renderer.sample_count());
   ImGui::Text("%.0f ms/sample", renderer.last_frame().milliseconds);
   ImGui::Text("%.0f Mrays/s", mrays_per_second);
}

void draw_display(Renderer& renderer) {
   ImGui::SeparatorText("Affichage");
   bool denoise = renderer.denoiser();
   if (ImGui::Checkbox("Denoiser OptiX", &denoise)) {
      renderer.set_denoiser(denoise);
   }
   int layer = static_cast<int>(renderer.display_layer());
   if (ImGui::Combo("AOV", &layer, "Beauty\0Albedo\0Normal\0")) {
      renderer.set_display_layer(static_cast<Renderer::DisplayLayer>(layer));
   }
   rt::DisplaySettings display = renderer.display();
   int tone_mapping = static_cast<int>(display.tone_mapping);
   bool changed = ImGui::Combo("Tone mapping", &tone_mapping, "AgX\0ACES\0Standard\0");
   changed |= ImGui::SliderFloat("Exposition (EV)", &display.exposure, -10.0f, 10.0f, "%.1f");
   if (changed) {
      display.tone_mapping = static_cast<rt::ToneMapping>(tone_mapping);
      renderer.set_display(display);
   }
}

} // namespace

Size Resolution::size(Size window) const {
   if (preset == kWindowPreset) {
      return window;
   }
   if (preset == kCustomPreset) {
      return custom;
   }
   return kPresets[preset].size;
}

bool draw_scene_panel(
   Interface& ui,
   Renderer& renderer,
   FlyCamera& camera,
   const rt::CameraSettings& initial_camera,
   Size window
) {
   ImGui::SetNextWindowPos({10.0f, 10.0f}, ImGuiCond_FirstUseEver);
   ImGui::Begin("Scène", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
   const bool locked = ui.final_render.locked();
   if (locked) {
      ImGui::TextDisabled("Figée pendant le rendu final");
   }
   ImGui::BeginDisabled(locked);
   draw_resolution(ui.resolution, window);
   const bool camera_changed = draw_camera(ui, camera, initial_camera);
   draw_sampling(ui, renderer);
   draw_environment(renderer);
   ImGui::EndDisabled();
   ImGui::End();
   return camera_changed;
}

RenderPanelResult draw_render_panel(Interface& ui, Renderer& renderer, Size image) {
   const ImGuiViewport* viewport = ImGui::GetMainViewport();
   ImGui::SetNextWindowPos(
      {viewport->WorkPos.x + viewport->WorkSize.x - 10.0f, viewport->WorkPos.y + 10.0f},
      ImGuiCond_FirstUseEver,
      {1.0f, 0.0f}
   );
   ImGui::Begin("Rendu", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
   draw_statistics(renderer, image);
   draw_display(renderer);

   RenderPanelResult result;
   FinalRender& final_render = ui.final_render;
   result.final_render = draw_final_render(
      final_render,
      renderer.sample_count(),
      Window::time(),
      ui.status
   );
   // Pendant le rendu final, l'export se fait depuis sa section
   if (!final_render.locked()) {
      ImGui::SeparatorText("Export");
      result.export_requested = ImGui::Button("Exporter (P)");
      if (!ui.status.empty()) {
         ImGui::TextUnformatted(ui.status.c_str());
      }
   }

   if (ImGui::CollapsingHeader("Commandes")) {
      ImGui::TextUnformatted(
         "ZQSD : se déplacer\n"
         "Espace, Ctrl : monter, descendre\n"
         "Maj : plus vite\n"
         "Clic droit : regarder\n"
         "Clic gauche : focus\n"
         "P : exporter\n"
         "Échap : quitter\n"
         "Inactives quand l'interface a le focus"
      );
   }
   ImGui::End();
   return result;
}
