#pragma once

#include "core/photon_map.h"
#include "core/photon_tracer.h"
#include "core/rng.h"
#include "core/scene_view.h"

#include <cstdint>
#include <vector>

namespace rt::test {

// Grille construite sur le CPU comme sur le GPU : comptage par alvéole, réservation des plages
// (dans l'ordre inverse, comme un ordre quelconque des atomiques), dispersion
struct CpuPhotonMap {
   std::vector<Photon> photons;
   std::vector<std::uint32_t> bucket_start;
   std::vector<std::uint32_t> bucket_size;
   PhotonMapView view;

   CpuPhotonMap(const std::vector<Photon>& input, std::uint32_t bucket_count, float radius):
      photons(input.size()),
      bucket_start(bucket_count, 0),
      bucket_size(bucket_count, 0) {
      const auto bucket = [&](const Photon& p) {
         return cell_hash(cell_of(p.position, 2.0f * radius), bucket_count);
      };
      for (const Photon& p : input) {
         ++bucket_size[bucket(p)];
      }
      std::uint32_t reserved = 0;
      for (std::uint32_t b = bucket_count; b-- > 0;) {
         bucket_start[b] = reserved;
         reserved += bucket_size[b];
      }
      std::vector<std::uint32_t> cursors = bucket_start;
      for (const Photon& p : input) {
         photons[cursors[bucket(p)]++] = p;
      }
      view = {
         .photons = photons.data(),
         .bucket_start = bucket_start.data(),
         .bucket_size = bucket_size.data(),
         .bucket_count = bucket_count,
         .radius = radius,
      };
   }
};

// Photons de caustique tracés sur le CPU, puissance divisée par le nombre émis
inline std::vector<Photon> trace_photons(const SceneView& view, int count, std::uint32_t seed) {
   Pcg32 rng(seed, 0);
   std::vector<Photon> photons;
   for (int i = 0; i < count; ++i) {
      Photon photon;
      if (trace_photon(view, emit_photon(view, rng), rng, photon)) {
         photon.power /= static_cast<float>(count);
         photons.push_back(photon);
      }
   }
   return photons;
}

} // namespace rt::test
