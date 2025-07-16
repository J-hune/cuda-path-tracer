#pragma once
#include "core/host_device.h"

#include <cstdint>

namespace rt {
// Générateur PCG32 (O'Neill, 2014) : état de 64 bits, flux sélectionnable, sortie de 32 bits.
// Référence : https://www.pcg-random.org
class Pcg32 {
public:
   RT_HOST_DEVICE static constexpr Pcg32 make_pixel_rng(
      std::uint32_t pixel_index,
      std::uint32_t sample_index
   ) {
      const std::uint64_t seed = mix64((std::uint64_t{pixel_index} << 32u) | sample_index);
      return {seed, pixel_index};
   }

   RT_HOST_DEVICE constexpr Pcg32(std::uint64_t seed, std::uint64_t stream):
      inc_((stream << 1u) | 1u) {
      step();
      state_ += seed;
      step();
   }

   RT_HOST_DEVICE constexpr std::uint32_t next_u32() {
      const std::uint64_t old_state = state_;
      step();

      const auto xorshifted = static_cast<std::uint32_t>(((old_state >> 18u) ^ old_state) >> 27u);
      const auto rot = static_cast<std::uint32_t>(old_state >> 59u);
      return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
   }

   // Nombre uniforme dans [0, 1[
   RT_HOST_DEVICE constexpr float next_float() {
      return static_cast<float>(next_u32() >> 8u) * 0x1p-24f;
   }

private:
   std::uint64_t state_ = 0;
   std::uint64_t inc_;

   RT_HOST_DEVICE constexpr void step() { state_ = state_ * 6364136223846793005ULL + inc_; }

   // Finaliseur de SplitMix64
   RT_HOST_DEVICE static constexpr std::uint64_t mix64(std::uint64_t x) {
      x ^= x >> 30u;
      x *= 0xbf58476d1ce4e5b9ULL;
      x ^= x >> 27u;
      x *= 0x94d049bb133111ebULL;
      x ^= x >> 31u;
      return x;
   }
};
} // namespace rt
