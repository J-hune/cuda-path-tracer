#pragma once

#include "core/photon_map.h"
#include "core/photon_tracer.h"
#include "core/scene_view.h"
#include "gpu/device_buffer.h"

#include <cstdint>

// Carte de photons de caustique sur le GPU, reconstruite à chaque passe : tracé des photons, puis
// rangement dans la grille de hachage (comptage par alvéole, réservation des plages, dispersion)
class PhotonMap {
public:
   explicit PhotonMap(std::uint32_t photon_count);

   // Émet photon_count photons (passe pass) et range ceux qui sont stockés
   void build(const rt::SceneView& scene, std::uint32_t pass, float radius);

   [[nodiscard]] rt::PhotonMapView view() const;
   [[nodiscard]] std::uint32_t photon_count() const noexcept { return photon_count_; }

private:
   std::uint32_t photon_count_;
   std::uint32_t bucket_count_;
   float radius_ = 0.0f;
   DeviceBuffer<rt::Photon> photons_;
   DeviceBuffer<rt::Photon> sorted_;
   DeviceBuffer<std::uint32_t> buckets_; // alvéole de chaque photon, kNotStored s'il est perdu
   DeviceBuffer<std::uint32_t> bucket_start_;
   DeviceBuffer<std::uint32_t> bucket_size_;
   DeviceBuffer<std::uint32_t> cursors_;
   DeviceBuffer<std::uint32_t> reserved_{1}; // places déjà réservées
};
