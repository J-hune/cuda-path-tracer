#pragma once

#include <cstdint>
#include <string>

// Rendu final : un nombre d'échantillons choisi, ou sans limite, rendu avec la caméra et les
// réglages figés. Il peut être continué une fois terminé.
struct FinalRender {
   enum class State : std::uint8_t { Off, Running, Finished };
   // Demande faite depuis l'interface
   enum class Action : std::uint8_t { None, Start, Stop, Cancel, Continue, Export, Close };

   State state = State::Off;
   int target_samples = 4096;
   bool unlimited = false;
   bool export_when_finished = true;

   double run_start = 0.0; // début de la session de rendu en cours, en secondes
   std::uint32_t run_start_samples = 0;
   double elapsed_before = 0.0; // durée des sessions terminées
   bool stop_requested = false; // arrêt à la fin de l'échantillon en cours
   bool exported = false;

   // Début d'un rendu, ou d'une nouvelle session après un rendu terminé (samples : échantillons
   // déjà accumulés)
   void start(double now, std::uint32_t samples);
   // Fin de la session : à la cible atteinte, la cible suivante proposée est le double
   void finish(double now, std::uint32_t samples);
   // La caméra et les réglages du rendu ne peuvent pas changer
   [[nodiscard]] bool locked() const { return state != State::Off; }
   // Durée totale de rendu, sessions précédentes comprises
   [[nodiscard]] double elapsed(double now) const;
   [[nodiscard]] bool reached(std::uint32_t samples) const;
};

// Section de l'interface : réglages, progression puis résultat du rendu. samples : échantillons
// accumulés, now : temps actuel en secondes, status : résultat du dernier export.
FinalRender::Action draw_final_render(
   FinalRender& render,
   std::uint32_t samples,
   double now,
   const std::string& status
);

// Durée lisible : « 12 s », « 3 min 05 s », « 1 h 02 min »
std::string format_duration(double seconds);
