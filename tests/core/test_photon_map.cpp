#include "core/photon_map.h"
#include "core/photon_tracer.h"
#include "core/rng.h"
#include "core/vec3.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <vector>

namespace {

// Grille construite sur le CPU comme sur le GPU : comptage par alvéole, réservation des plages
// (dans l'ordre inverse, comme un ordre quelconque des atomiques), dispersion
struct CpuPhotonMap {
   std::vector<rt::Photon> photons;
   std::vector<std::uint32_t> bucket_start;
   std::vector<std::uint32_t> bucket_size;
   rt::PhotonMapView view;

   CpuPhotonMap(const std::vector<rt::Photon>& input, std::uint32_t bucket_count, float radius):
      photons(input.size()),
      bucket_start(bucket_count, 0),
      bucket_size(bucket_count, 0) {
      const auto bucket = [&](const rt::Photon& p) {
         return rt::cell_hash(rt::cell_of(p.position, 2.0f * radius), bucket_count);
      };
      for (const rt::Photon& p : input) {
         ++bucket_size[bucket(p)];
      }
      std::uint32_t reserved = 0;
      for (std::uint32_t b = bucket_count; b-- > 0;) {
         bucket_start[b] = reserved;
         reserved += bucket_size[b];
      }
      std::vector<std::uint32_t> cursors = bucket_start;
      for (const rt::Photon& p : input) {
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

} // namespace

TEST_CASE("grille de photons : mêmes voisins qu'une recherche exhaustive", "[photon_map]") {
   rt::Pcg32 rng(5, 0);
   std::vector<rt::Photon> photons;
   for (int i = 0; i < 20'000; ++i) {
      // Coordonnées négatives et positives, photon identifié par sa puissance
      const rt::Vec3 p{
         (2.0f * rng.next_float()) - 1.0f,
         (2.0f * rng.next_float()) - 1.0f,
         (0.2f * rng.next_float()) - 0.1f,
      };
      photons.push_back({
         .position = p,
         .power = {static_cast<float>(i), 0, 0},
         .direction = {0, 0, 1},
         .normal = {0, 0, 1},
      });
   }
   constexpr float kRadius = 0.05f;

   // Peu d'alvéoles : beaucoup de cellules différentes se partagent la même
   for (const std::uint32_t buckets : {16u, 4096u}) {
      const CpuPhotonMap map(photons, buckets, kRadius);
      for (int q = 0; q < 500; ++q) {
         const rt::Vec3 x{
            (2.0f * rng.next_float()) - 1.0f,
            (2.0f * rng.next_float()) - 1.0f,
            (0.2f * rng.next_float()) - 0.1f,
         };
         std::vector<float> found;
         rt::for_each_photon_near(map.view, x, [&](const rt::Photon& p) {
            found.push_back(p.power.x);
         });
         std::vector<float> expected;
         for (const rt::Photon& p : photons) {
            if (rt::length_squared(p.position - x) < kRadius * kRadius) {
               expected.push_back(p.power.x);
            }
         }
         std::ranges::sort(found);
         std::ranges::sort(expected);
         CHECK(found == expected);
      }
   }
}
