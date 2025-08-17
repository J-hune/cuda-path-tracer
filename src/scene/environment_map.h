#pragma once

#include "core/environment.h"
#include "core/vec3.h"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace rt {

// HDRI chargée en mémoire, avec ses fonctions de répartition (voir EnvironmentMap)
struct EnvironmentImage {
   std::uint32_t width = 0;
   std::uint32_t height = 0;
   std::vector<Vec3> texels;
   std::vector<float> marginal_cdf;
   std::vector<float> conditional_cdf;
   float average_luminance = 0.0f;
   float strength = 1.0f;
   float rotation = 0.0f; // radians
   std::string source;    // fichier d'origine

   [[nodiscard]] bool empty() const { return texels.empty(); }

   // Vue sur ces tableaux (CPU)
   [[nodiscard]] EnvironmentMap view() const;
};

// Lit une HDRI (.exr, ou .hdr) et construit ses fonctions de répartition
std::expected<EnvironmentImage, std::string> load_environment(
   const std::filesystem::path& path,
   float strength,
   float rotation
);

// Fonctions de répartition d'une image déjà chargée (texels, width, height) : chaque texel est
// tiré selon sa luminance × sin θ, l'aire qu'il couvre sur la sphère
void build_environment_distribution(EnvironmentImage& image);

} // namespace rt
