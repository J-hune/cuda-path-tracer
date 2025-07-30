#pragma once

#include "gpu/renderer.h"

#include <cstdint>
#include <string>

// Manifeste de rendu : décrit comment une image exportée a été produite (machine, rendu, réglages),
// pour que chaque image et chaque mesure de performance porte son contexte.
struct RenderManifest {
   std::string date; // heure locale, ISO 8601
   Renderer::DeviceInfo device;

   std::string scene;
   int width = 0;
   int height = 0;
   std::uint32_t samples_per_pixel = 0;
   double gpu_milliseconds = 0.0;
   std::uint64_t rays = 0;

   bool mis = true;
   float firefly_clamp = 0.0f; // 0 : désactivé
   int max_bounces = 0;
   int russian_roulette_start = 0;
   std::string bvh;
   std::string tone_mapping;
   float exposure = 1.0f;
};

// Heure locale actuelle au format ISO 8601, avec le décalage horaire (2025-07-14T18:03:12+0200)
std::string current_local_time();

// Écrit le manifeste en JSON
bool write_manifest(const std::string& path, const RenderManifest& manifest);
