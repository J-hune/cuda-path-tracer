#pragma once

#include "app/window.h"

#include <cstdint>
#include <string>

// Rendu final : une résolution et un nombre d'échantillons choisis, rendus jusqu'au bout avec la
// caméra figée, puis exportés
struct FinalRender {
   enum class State : std::uint8_t { Off, Running, Finished };
   // Demande faite depuis l'interface
   enum class Action : std::uint8_t { None, Start, Stop, Cancel, Export, Close };

   State state = State::Off;
   int resolution = 1; // préréglage choisi
   Size custom{1920, 1080};
   int target_samples = 4096;
   bool export_when_finished = true;

   Size size; // du rendu en cours ou terminé
   double start_time = 0.0;
   double end_time = 0.0;
   bool stop_requested = false; // arrêt à la fin de l'échantillon en cours
   bool exported = false;

   // Taille choisie dans l'interface, window : taille actuelle de l'aperçu
   [[nodiscard]] Size chosen_size(Size window) const;
   // La caméra et les réglages du rendu ne peuvent pas changer
   [[nodiscard]] bool locked() const { return state != State::Off; }
};

// Section de l'interface : réglages, progression puis résultat du rendu. samples : échantillons
// accumulés, now : temps actuel en secondes, status : résultat du dernier export.
FinalRender::Action draw_final_render(
   FinalRender& render,
   std::uint32_t samples,
   Size window,
   double now,
   const std::string& status
);

// Durée lisible : « 12 s », « 3 min 05 s », « 1 h 02 min »
std::string format_duration(double seconds);
