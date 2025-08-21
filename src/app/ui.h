#pragma once

#include "app/final_render.h"
#include "app/fly_camera.h"
#include "app/window.h"
#include "core/camera.h"
#include "core/path_tracer.h"
#include "gpu/renderer.h"

#include <string>

// Taille de l'image rendue : un préréglage, la taille de la fenêtre ou une taille personnalisée.
// L'image est affichée à l'échelle de la fenêtre.
struct Resolution {
   int preset = 0; // taille de la fenêtre
   Size custom{1920, 1080};

   [[nodiscard]] Size size(Size window) const;
};

// État de l'interface qui n'appartient ni au renderer ni à la caméra
struct Interface {
   Resolution resolution;
   float firefly_threshold = rt::kDefaultFireflyClamp;
   float scene_extent = 1.0f; // diagonale de la scène
   float move_speed = 1.0f;   // unités de la scène par seconde
   FinalRender final_render;
   std::string status; // résultat du dernier export
};

// Panneau de gauche : ce qui définit l'image (résolution, caméra, échantillonnage, environnement),
// figé pendant le rendu final. window : taille de la fenêtre. Renvoie vrai si la caméra a changé.
bool draw_scene_panel(
   Interface& ui,
   Renderer& renderer,
   FlyCamera& camera,
   const rt::CameraSettings& initial_camera,
   Size window
);

struct RenderPanelResult {
   FinalRender::Action final_render = FinalRender::Action::None;
   bool export_requested = false;
};

// Panneau de droite : statistiques, affichage, rendu final et export. image : taille de l'image.
RenderPanelResult draw_render_panel(Interface& ui, Renderer& renderer, Size image);
