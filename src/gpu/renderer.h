#pragma once

#include "core/camera.h"
#include "core/tonemap.h"
#include "core/vec3.h"
#include "scene/bvh_builder.h"
#include "scene/gltf_loader.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Rendu GPU d'une scène. La scène est envoyée sur le GPU une seule fois, à la construction.
// Les types CUDA restent cachés dans renderer.cu (idiome pimpl) : ce fichier reste du C++ pur.
class Renderer {
public:
   // Couche affichée dans la fenêtre
   enum class DisplayLayer : std::uint8_t { Color, Albedo, Normal };

   // Mesures de la dernière image : temps GPU du rendu et nombre de rayons lancés
   struct FrameStats {
      float milliseconds = 0.0f;
      std::uint64_t rays = 0;
   };

   // Cumul depuis le dernier redémarrage de l'accumulation (reset, resize, changement de réglage)
   struct AccumulatedStats {
      double milliseconds = 0.0;
      std::uint64_t rays = 0;
   };

   // Carte graphique utilisée
   struct DeviceInfo {
      std::string name;
      int compute_capability_major = 0;
      int compute_capability_minor = 0;
      std::size_t memory_bytes = 0;
      int driver_version = 0;  // ex. 13030 pour CUDA 13.3
      int runtime_version = 0; // version du runtime CUDA du programme, même format
   };

   // Paramètres des caustiques par photons
   struct CausticSettings {
      std::uint32_t photons_per_pass = 0;
      float initial_radius = 0.0f;        // rayon maximal
      float initial_radius_pixels = 0.0f; // vu depuis la caméra
      float alpha = 0.0f;                 // vitesse de diminution du rayon
   };

   Renderer(const rt::Scene& scene, const rt::Bvh& bvh, int width, int height);
   ~Renderer();
   Renderer(const Renderer&) = delete;
   Renderer& operator=(const Renderer&) = delete;
   Renderer(Renderer&&) noexcept;
   Renderer& operator=(Renderer&&) noexcept;

   // Pixel buffer OpenGL (RGBA8, width × height) dans lequel les images sont écrites
   void attach_pixel_buffer(unsigned int gl_buffer);
   // Libère l'enregistrement CUDA du pixel buffer (avant qu'OpenGL ne le réalloue)
   void detach_pixel_buffer();

   // Change la taille de l'image et repart de zéro
   void resize(int width, int height);

   // Ajoute un échantillon par pixel, puis écrit la moyenne accumulée dans le pixel buffer
   void render(const rt::Camera& camera);
   // Avance l'échantillon en cours pendant environ milliseconds (temps GPU), par tranches de
   // pixels : l'interface reste fluide même quand un échantillon complet prend des secondes.
   // Renvoie vrai quand l'échantillon est terminé et affiché. La caméra ne doit pas changer en
   // cours d'échantillon.
   bool render_for(const rt::Camera& camera, float milliseconds);
   // Réécrit l'image affichée sans ajouter d'échantillon (réglages d'affichage, débruitage)
   void refresh_display();

   // Repart de zéro (à appeler quand la caméra ou la scène change)
   void reset();

   // Active ou désactive le MIS (sinon : éclairage direct seul pour les lampes), puis repart de
   // zéro
   void set_mis(bool enabled);
   [[nodiscard]] bool mis() const;

   // Exposition et tone mapping de l'image affichée, sans effet sur l'accumulation
   void set_display(rt::DisplaySettings display);
   [[nodiscard]] rt::DisplaySettings display() const;

   void set_display_layer(DisplayLayer layer);
   [[nodiscard]] DisplayLayer display_layer() const;

   // Écrêtage des fireflies (biaisé), 0 : désactivé. Repart de zéro
   void set_firefly_clamp(float max_contribution);
   [[nodiscard]] float firefly_clamp() const;

   // HDRI : intensité et rotation autour de l'axe vertical (radians), puis repart de zéro
   [[nodiscard]] bool has_environment_map() const;
   void set_environment(float strength, float rotation);
   [[nodiscard]] float environment_strength() const;
   [[nodiscard]] float environment_rotation() const;

   // Caustiques par photons, puis repart de zéro
   void set_caustics(bool enabled);
   [[nodiscard]] bool caustics() const;
   [[nodiscard]] CausticSettings caustic_settings() const;

   // Débruiteur OptiX sur la couche couleur affichée, relancé à intervalles croissants selon son
   // coût
   void set_denoiser(bool enabled);
   [[nodiscard]] bool denoiser() const;

   // Nombre d'échantillons par pixel accumulés depuis le dernier reset
   [[nodiscard]] std::uint32_t sample_count() const;

   [[nodiscard]] FrameStats last_frame() const;
   [[nodiscard]] AccumulatedStats accumulated() const;
   [[nodiscard]] static DeviceInfo device_info();

   // Moyenne accumulée d'une couche (couleur linéaire par défaut), copiée sur le CPU
   [[nodiscard]] std::vector<rt::Vec3> image(DisplayLayer layer = DisplayLayer::Color) const;
   // Couleur accumulée actuelle, débruitée (vide si rien n'est accumulé)
   [[nodiscard]] std::vector<rt::Vec3> denoised_image();

private:
   struct Impl;
   std::unique_ptr<Impl> impl_;
};
