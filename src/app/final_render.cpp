#include "app/final_render.h"

#include "app/window.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
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
   {.name = "720p", .size = {1280, 720}},
   {.name = "1080p", .size = {1920, 1080}},
   {.name = "1440p", .size = {2560, 1440}},
   {.name = "4K", .size = {3840, 2160}},
   {.name = "Fenêtre", .size = {}},
   {.name = "Personnalisée", .size = {}},
}};
constexpr int kWindowPreset = 4;
constexpr int kCustomPreset = 5;
constexpr int kMinSide = 16;
constexpr int kMaxSide = 4096; // au-delà, les tampons et le débruiteur dépassent la mémoire du GPU
constexpr int kMaxSamples = 1 << 20;

std::string preset_label(int index, Size window) {
   if (index == kCustomPreset) {
      return kPresets[kCustomPreset].name;
   }
   const Size size = index == kWindowPreset ? window : kPresets[index].size;
   return std::format("{} × {} ({})", size.width, size.height, kPresets[index].name);
}

// Fenêtre de confirmation centrée. Renvoie vrai si l'action est confirmée.
bool confirm(const char* title, const char* message, const char* yes, const char* no) {
   bool confirmed = false;
   ImGui::SetNextWindowPos(
      ImGui::GetMainViewport()->GetCenter(),
      ImGuiCond_Appearing,
      {0.5f, 0.5f}
   );
   if (ImGui::BeginPopupModal(title, nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
      ImGui::TextUnformatted(message);
      ImGui::Spacing();
      if (ImGui::Button(yes)) {
         confirmed = true;
         ImGui::CloseCurrentPopup();
      }
      ImGui::SameLine();
      if (ImGui::Button(no)) {
         ImGui::CloseCurrentPopup();
      }
      ImGui::EndPopup();
   }
   return confirmed;
}

FinalRender::Action draw_settings(FinalRender& render, Size window) {
   if (ImGui::BeginCombo("Résolution", preset_label(render.resolution, window).c_str())) {
      for (int i = 0; std::cmp_less(i, kPresets.size()); ++i) {
         const bool selected = i == render.resolution;
         if (ImGui::Selectable(preset_label(i, window).c_str(), selected)) {
            render.resolution = i;
         }
         if (selected) {
            ImGui::SetItemDefaultFocus();
         }
      }
      ImGui::EndCombo();
   }
   if (render.resolution == kCustomPreset) {
      std::array<int, 2> size{render.custom.width, render.custom.height};
      if (ImGui::InputInt2("Largeur, hauteur", size.data())) {
         render.custom = {
            .width = std::clamp(size[0], kMinSide, kMaxSide),
            .height = std::clamp(size[1], kMinSide, kMaxSide),
         };
      }
   }
   if (ImGui::InputInt("Samples", &render.target_samples, 256, 1024)) {
      render.target_samples = std::clamp(render.target_samples, 1, kMaxSamples);
   }
   ImGui::TextDisabled("4096 : image propre, 8192 : rendu final");
   ImGui::Checkbox("Exporter à la fin", &render.export_when_finished);
   if (ImGui::Button("Lancer le rendu", {-FLT_MIN, 0.0f})) {
      return FinalRender::Action::Start;
   }
   return FinalRender::Action::None;
}

FinalRender::Action draw_progress(const FinalRender& render, std::uint32_t samples, double now) {
   const auto target = static_cast<std::uint32_t>(render.target_samples);
   ImGui::Text("%d × %d pixels", render.size.width, render.size.height);
   const float fraction = std::fmin(1.0f, static_cast<float>(samples) / static_cast<float>(target));
   const std::string overlay = std::format("{} / {} samples", samples, target);
   ImGui::ProgressBar(fraction, {-FLT_MIN, 0.0f}, overlay.c_str());

   const double elapsed = now - render.start_time;
   ImGui::Text("Écoulé : %s", format_duration(elapsed).c_str());
   if (samples > 0) {
      const double remaining = elapsed / samples *
                               static_cast<double>(target - std::min(samples, target));
      ImGui::Text("Restant : environ %s", format_duration(remaining).c_str());
   } else {
      ImGui::TextUnformatted("Restant : estimé après le premier sample");
   }

   FinalRender::Action action = FinalRender::Action::None;
   if (render.stop_requested) {
      ImGui::TextDisabled("Arrêt à la fin du sample en cours…");
   }
   ImGui::BeginDisabled(render.stop_requested);
   if (ImGui::Button("Arrêter")) {
      action = FinalRender::Action::Stop;
   }
   ImGui::SetItemTooltip("Termine le sample en cours et garde l'image");
   ImGui::EndDisabled();
   ImGui::SameLine();
   if (ImGui::Button("Annuler")) {
      ImGui::OpenPopup("Annuler le rendu ?");
   }
   ImGui::SetItemTooltip("Abandonne le rendu et revient à l'aperçu");
   if (confirm(
          "Annuler le rendu ?",
          "L'image en cours sera perdue.",
          "Annuler le rendu",
          "Continuer"
       )) {
      action = FinalRender::Action::Cancel;
   }
   return action;
}

FinalRender::Action draw_result(
   const FinalRender& render,
   std::uint32_t samples,
   const std::string& status
) {
   ImGui::Text("%d × %d pixels", render.size.width, render.size.height);
   ImGui::Text(
      "Terminé : %u samples en %s",
      samples,
      format_duration(render.end_time - render.start_time).c_str()
   );
   FinalRender::Action action = FinalRender::Action::None;
   if (ImGui::Button("Exporter (P)")) {
      action = FinalRender::Action::Export;
   }
   ImGui::SameLine();
   if (ImGui::Button("Retour à l'aperçu")) {
      if (render.exported) {
         action = FinalRender::Action::Close;
      } else {
         ImGui::OpenPopup("Revenir à l'aperçu ?");
      }
   }
   if (!status.empty()) {
      ImGui::TextUnformatted(status.c_str());
   }
   if (confirm(
          "Revenir à l'aperçu ?",
          "L'image n'a pas été exportée, elle sera perdue.",
          "Revenir à l'aperçu",
          "Rester"
       )) {
      action = FinalRender::Action::Close;
   }
   return action;
}

} // namespace

Size FinalRender::chosen_size(Size window) const {
   if (resolution == kWindowPreset) {
      return window;
   }
   if (resolution == kCustomPreset) {
      return custom;
   }
   return kPresets[resolution].size;
}

FinalRender::Action draw_final_render(
   FinalRender& render,
   std::uint32_t samples,
   Size window,
   double now,
   const std::string& status
) {
   ImGui::SeparatorText("Rendu final");
   FinalRender::Action action = FinalRender::Action::None;
   switch (render.state) {
      case FinalRender::State::Off:
         action = draw_settings(render, window);
         break;
      case FinalRender::State::Running:
         action = draw_progress(render, samples, now);
         break;
      case FinalRender::State::Finished:
         action = draw_result(render, samples, status);
         break;
   }
   return action;
}

std::string format_duration(double seconds) {
   const auto total = static_cast<long long>(std::lround(std::fmax(seconds, 0.0)));
   const long long hours = total / 3600;
   const long long minutes = (total / 60) % 60;
   if (hours > 0) {
      return std::format("{} h {:02} min", hours, minutes);
   }
   if (minutes > 0) {
      return std::format("{} min {:02} s", minutes, total % 60);
   }
   return std::format("{} s", total);
}
