#include "app/final_render.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <format>
#include <imgui.h>
#include <string>
#include <utility>

namespace {

constexpr int kMaxSamples = 1 << 20;

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

// Nombre d'échantillons visé, ou rendu sans limite jusqu'à l'arrêt
void draw_target(FinalRender& render, const char* label) {
   ImGui::BeginDisabled(render.unlimited);
   if (ImGui::InputInt(label, &render.target_samples, 256, 1024)) {
      render.target_samples = std::clamp(render.target_samples, 1, kMaxSamples);
   }
   ImGui::EndDisabled();
   ImGui::Checkbox("Sans limite", &render.unlimited);
   ImGui::SetItemTooltip("Le rendu continue jusqu'à ce qu'on l'arrête");
}

FinalRender::Action draw_settings(FinalRender& render) {
   draw_target(render, "Samples");
   ImGui::TextDisabled("4096 : image propre, 8192 : rendu final");
   ImGui::Checkbox("Exporter à la fin", &render.export_when_finished);
   if (ImGui::Button("Lancer le rendu", {-FLT_MIN, 0.0f})) {
      return FinalRender::Action::Start;
   }
   return FinalRender::Action::None;
}

FinalRender::Action draw_progress(const FinalRender& render, std::uint32_t samples, double now) {
   if (render.unlimited) {
      ImGui::Text("%u samples, sans limite", samples);
   } else {
      const auto target = static_cast<std::uint32_t>(render.target_samples);
      const float fraction = std::fmin(
         1.0f,
         static_cast<float>(samples) / static_cast<float>(target)
      );
      const std::string overlay = std::format("{} / {} samples", samples, target);
      ImGui::ProgressBar(fraction, {-FLT_MIN, 0.0f}, overlay.c_str());
   }

   ImGui::Text("Écoulé : %s", format_duration(render.elapsed(now)).c_str());
   if (!render.unlimited) {
      // Vitesse de la session en cours : celle d'avant a pu avoir d'autres réglages
      const std::uint32_t done = samples - render.run_start_samples;
      const auto target = static_cast<std::uint32_t>(render.target_samples);
      if (done > 0) {
         const double remaining = (now - render.run_start) / done *
                                  static_cast<double>(target - std::min(samples, target));
         ImGui::Text("Restant : environ %s", format_duration(remaining).c_str());
      } else {
         ImGui::TextUnformatted("Restant : estimé après le premier sample");
      }
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
   FinalRender& render,
   std::uint32_t samples,
   double now,
   const std::string& status
) {
   ImGui::Text("Terminé : %u samples en %s", samples, format_duration(render.elapsed(now)).c_str());
   FinalRender::Action action = FinalRender::Action::None;
   draw_target(render, "Continuer jusqu'à");
   const bool can_continue = render.unlimited || std::cmp_greater(render.target_samples, samples);
   ImGui::BeginDisabled(!can_continue);
   if (ImGui::Button("Continuer le rendu", {-FLT_MIN, 0.0f})) {
      action = FinalRender::Action::Continue;
   }
   ImGui::EndDisabled();
   if (ImGui::Button("Exporter (P)")) {
      action = FinalRender::Action::Export;
   }
   ImGui::SameLine();
   if (ImGui::Button("Retour à l'aperçu")) {
      action = FinalRender::Action::Close;
   }
   ImGui::SetItemTooltip("Débloque la caméra et les réglages, l'image reste affichée");
   if (!status.empty()) {
      ImGui::TextUnformatted(status.c_str());
   }
   return action;
}

} // namespace

void FinalRender::start(double now, std::uint32_t samples) {
   if (state == State::Off) {
      elapsed_before = 0.0;
   }
   state = State::Running;
   run_start = now;
   run_start_samples = samples;
   stop_requested = false;
   exported = false;
}

void FinalRender::finish(double now, std::uint32_t samples) {
   elapsed_before += now - run_start;
   if (reached(samples)) {
      target_samples = static_cast<int>(std::min(2 * samples, std::uint32_t{kMaxSamples}));
   }
   state = State::Finished;
}

double FinalRender::elapsed(double now) const {
   return state == State::Running ? elapsed_before + (now - run_start) : elapsed_before;
}

bool FinalRender::reached(std::uint32_t samples) const {
   return !unlimited && std::cmp_greater_equal(samples, target_samples);
}

FinalRender::Action draw_final_render(
   FinalRender& render,
   std::uint32_t samples,
   double now,
   const std::string& status
) {
   ImGui::SeparatorText("Rendu final");
   switch (render.state) {
      case FinalRender::State::Off:
         return draw_settings(render);
      case FinalRender::State::Running:
         return draw_progress(render, samples, now);
      case FinalRender::State::Finished:
         break;
   }
   return draw_result(render, samples, now, status);
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
