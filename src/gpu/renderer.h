#pragma once

#include "core/camera.h"
#include "core/vec3.h"
#include "scene/bvh_builder.h"
#include "scene/gltf_loader.h"

#include <cstdint>
#include <memory>
#include <vector>

// Rendu GPU d'une scène. La scène est envoyée sur le GPU une seule fois, à la construction.
// Les types CUDA restent cachés dans renderer.cu (idiome pimpl) : ce fichier reste du C++ pur.
class Renderer {
public:
   // Mesures de la dernière image : temps GPU du rendu et nombre de rayons lancés
   struct FrameStats {
      float milliseconds = 0.0f;
      std::uint64_t rays = 0;
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

   // Nombre d'échantillons par pixel accumulés depuis le dernier reset
   [[nodiscard]] std::uint32_t sample_count() const;

   [[nodiscard]] FrameStats last_frame() const;

   // Image accumulée (moyenne linéaire), copiée sur le CPU
   [[nodiscard]] std::vector<rt::Vec3> image() const;

private:
   struct Impl;
   std::unique_ptr<Impl> impl_;
};
