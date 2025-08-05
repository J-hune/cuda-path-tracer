#pragma once

#include "core/host_device.h"
#include "core/photon_tracer.h"
#include "core/vec3.h"

#include <cmath>
#include <cstdint>

namespace rt {

// Coordonnées entières d'une cellule de la grille
struct Cell {
   int x = 0;
   int y = 0;
   int z = 0;

   constexpr bool operator==(const Cell&) const = default;
};

// Photons rangés par cellule d'une grille de hachage : les photons de l'alvéole h occupent
// bucket_count[h] places consécutives à partir de photons[bucket_start[h]]. Les cellules mesurent
// 2 × radius : une sphère de rayon radius en touche au plus 2 × 2 × 2.
struct PhotonMapView {
   const Photon* photons = nullptr;
   const std::uint32_t* bucket_start = nullptr;
   const std::uint32_t* bucket_size = nullptr;
   std::uint32_t bucket_count = 0; // puissance de 2
   float radius = 0.0f;
};

RT_HOST_DEVICE inline Cell cell_of(Vec3 p, float cell_size) {
   return {
      static_cast<int>(std::floor(p.x / cell_size)),
      static_cast<int>(std::floor(p.y / cell_size)),
      static_cast<int>(std::floor(p.z / cell_size)),
   };
}

// Teschner et al., « Optimized Spatial Hashing for Collision Detection of Deformable Objects »,
// 2003
RT_HOST_DEVICE inline std::uint32_t cell_hash(Cell cell, std::uint32_t bucket_count) {
   const std::uint32_t h = (static_cast<std::uint32_t>(cell.x) * 73856093u) ^
                           (static_cast<std::uint32_t>(cell.y) * 19349663u) ^
                           (static_cast<std::uint32_t>(cell.z) * 83492791u);
   return h & (bucket_count - 1u);
}

// Appelle visit(photon) pour chaque photon à moins de map.radius de x
template <typename Visit>
RT_HOST_DEVICE void for_each_photon_near(const PhotonMapView& map, Vec3 x, Visit&& visit) {
   if (map.bucket_count == 0) {
      return;
   }
   const float cell_size = 2.0f * map.radius;
   const float radius2 = map.radius * map.radius;
   const Cell low = cell_of(x - Vec3{map.radius, map.radius, map.radius}, cell_size);
   const Cell high = cell_of(x + Vec3{map.radius, map.radius, map.radius}, cell_size);
   for (int cz = low.z; cz <= high.z; ++cz) {
      for (int cy = low.y; cy <= high.y; ++cy) {
         for (int cx = low.x; cx <= high.x; ++cx) {
            const Cell cell{cx, cy, cz};
            const std::uint32_t h = cell_hash(cell, map.bucket_count);
            const std::uint32_t end = map.bucket_start[h] + map.bucket_size[h];
            for (std::uint32_t i = map.bucket_start[h]; i < end; ++i) {
               const Photon& photon = map.photons[i];
               // Deux cellules peuvent partager une alvéole : on ne garde que celle du photon
               if (length_squared(photon.position - x) < radius2 &&
                   cell_of(photon.position, cell_size) == cell) {
                  visit(photon);
               }
            }
         }
      }
   }
}

} // namespace rt
