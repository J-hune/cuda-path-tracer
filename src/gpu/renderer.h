#pragma once

#include "core/camera.h"
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

   // Repart de zéro (à appeler quand la caméra ou la scène change)
   void reset();

   // Active ou désactive le MIS (sinon : éclairage direct seul pour les lampes), puis repart de
   // zéro
   void set_mis(bool enabled);
   [[nodiscard]] bool mis() const;

   void set_display_layer(DisplayLayer layer);
   [[nodiscard]] DisplayLayer display_layer() const;

   // Écrêtage des fireflies (biaisé), 0 : désactivé. Repart de zéro
   void set_firefly_clamp(float max_contribution);
   [[nodiscard]] float firefly_clamp() const;

   // Nombre d'échantillons par pixel accumulés depuis le dernier reset
   [[nodiscard]] std::uint32_t sample_count() const;

   [[nodiscard]] FrameStats last_frame() const;
   [[nodiscard]] AccumulatedStats accumulated() const;
   [[nodiscard]] static DeviceInfo device_info();

   // Moyenne accumulée d'une couche (couleur linéaire par défaut), copiée sur le CPU
   [[nodiscard]] std::vector<rt::Vec3> image(DisplayLayer layer = DisplayLayer::Color) const;

private:
   struct Impl;
   std::unique_ptr<Impl> impl_;
};
