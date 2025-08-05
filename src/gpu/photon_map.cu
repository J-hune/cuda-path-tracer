#include "core/photon_map.h"
#include "core/photon_tracer.h"
#include "core/rng.h"
#include "core/scene_view.h"
#include "gpu/cuda_check.h"
#include "gpu/device_buffer.h"
#include "gpu/photon_map.h"

#include <cuda_runtime.h>

#include <bit>
#include <cstdint>

namespace {

constexpr std::uint32_t kNotStored = 0xFFFFFFFFu;

int block_count(std::uint32_t n) {
   return static_cast<int>((n + 255) / 256);
}

__global__ void trace_kernel(
   rt::SceneView scene,
   std::uint32_t pass,
   std::uint32_t photon_count,
   std::uint32_t bucket_count,
   float cell_size,
   rt::Photon* photons,
   std::uint32_t* buckets
) {
   const std::uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
   if (i >= photon_count) {
      return;
   }
   rt::Pcg32 rng = rt::Pcg32::make_photon_rng(i, pass);
   rt::Photon photon;
   if (!rt::trace_photon(scene, rt::emit_photon(scene, rng), rng, photon)) {
      buckets[i] = kNotStored;
      return;
   }
   photon.power /= static_cast<float>(photon_count);
   photons[i] = photon;
   buckets[i] = rt::cell_hash(rt::cell_of(photon.position, cell_size), bucket_count);
}

__global__ void count_kernel(
   const std::uint32_t* buckets,
   std::uint32_t photon_count,
   std::uint32_t* counts
) {
   const std::uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
   if (i < photon_count && buckets[i] != kNotStored) {
      atomicAdd(&counts[buckets[i]], 1u);
   }
}

// Chaque alvéole non vide réserve sa plage. L'ordre des plages est quelconque : seule compte leur
// contiguïté, ce qui évite une somme préfixe.
__global__ void reserve_kernel(
   const std::uint32_t* sizes,
   std::uint32_t bucket_count,
   std::uint32_t* reserved,
   std::uint32_t* starts
) {
   const std::uint32_t b = (blockIdx.x * blockDim.x) + threadIdx.x;
   if (b < bucket_count && sizes[b] > 0) {
      starts[b] = atomicAdd(reserved, sizes[b]);
   }
}

__global__ void scatter_kernel(
   const rt::Photon* photons,
   const std::uint32_t* buckets,
   std::uint32_t photon_count,
   const std::uint32_t* starts,
   std::uint32_t* cursors,
   rt::Photon* sorted
) {
   const std::uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
   if (i >= photon_count || buckets[i] == kNotStored) {
      return;
   }
   const std::uint32_t b = buckets[i];
   sorted[starts[b] + atomicAdd(&cursors[b], 1u)] = photons[i];
}

} // namespace

PhotonMap::PhotonMap(std::uint32_t photon_count):
   photon_count_(photon_count),
   bucket_count_(std::bit_ceil(photon_count)),
   photons_(photon_count),
   sorted_(photon_count),
   buckets_(photon_count),
   bucket_start_(bucket_count_),
   bucket_size_(bucket_count_),
   cursors_(bucket_count_) {}

void PhotonMap::build(const rt::SceneView& scene, std::uint32_t pass, float radius) {
   radius_ = radius;
   const float cell_size = 2.0f * radius;
   trace_kernel<<<block_count(photon_count_), 256>>>(
      scene,
      pass,
      photon_count_,
      bucket_count_,
      cell_size,
      photons_.data(),
      buckets_.data()
   );
   RT_CUDA_CHECK_LAUNCH();

   RT_CUDA_CHECK(cudaMemset(bucket_size_.data(), 0, bucket_size_.size_bytes()));
   RT_CUDA_CHECK(cudaMemset(cursors_.data(), 0, cursors_.size_bytes()));
   RT_CUDA_CHECK(cudaMemset(reserved_.data(), 0, reserved_.size_bytes()));
   count_kernel<<<block_count(photon_count_), 256>>>(
      buckets_.data(),
      photon_count_,
      bucket_size_.data()
   );
   RT_CUDA_CHECK_LAUNCH();
   reserve_kernel<<<block_count(bucket_count_), 256>>>(
      bucket_size_.data(),
      bucket_count_,
      reserved_.data(),
      bucket_start_.data()
   );
   RT_CUDA_CHECK_LAUNCH();

   scatter_kernel<<<block_count(photon_count_), 256>>>(
      photons_.data(),
      buckets_.data(),
      photon_count_,
      bucket_start_.data(),
      cursors_.data(),
      sorted_.data()
   );
   RT_CUDA_CHECK_LAUNCH();
}

rt::PhotonMapView PhotonMap::view() const {
   return {
      .photons = sorted_.data(),
      .bucket_start = bucket_start_.data(),
      .bucket_size = bucket_size_.data(),
      .bucket_count = bucket_count_,
      .radius = radius_,
   };
}
