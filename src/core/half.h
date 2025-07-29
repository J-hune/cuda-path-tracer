#pragma once

#include <bit>
#include <cstdint>

namespace rt {

// Conversion d'un float 32 bits en demi-précision IEEE 754 (16 bits), arrondi au plus proche
// (à égalité, vers le pair). Algorithme de Fabian Giesen (« float_to_half_fast3_rtne »).
// Les valeurs trop grandes deviennent ±infini, NaN reste NaN.
inline std::uint16_t float_to_half_bits(float value) {
   auto bits = std::bit_cast<std::uint32_t>(value);
   const auto sign = static_cast<std::uint16_t>((bits >> 16U) & 0x8000U);
   bits &= 0x7FFFFFFFU;

   // Infini, NaN, ou au moins 2¹⁶ : hors de la demi-précision (les valeurs entre 65504 et 2¹⁶
   // sont arrondies plus bas, vers 65504 ou l'infini)
   if (bits >= 0x47800000U) {
      const bool is_nan = bits > 0x7F800000U;
      return sign | static_cast<std::uint16_t>(is_nan ? 0x7E00U : 0x7C00U);
   }

   // Sous-normaux et zéro (< 2⁻¹⁴) : l'addition en float de 0,5 aligne la mantisse et arrondit
   if (bits < 0x38800000U) {
      const float shifted = std::bit_cast<float>(bits) + 0.5f;
      return sign | static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(shifted) - 0x3F000000U);
   }

   // Normaux : changement de biais d'exposant (127 -> 15) et arrondi au plus proche pair
   const std::uint32_t mantissa_odd = (bits >> 13U) & 1U;
   bits += 0xC8000FFFU + mantissa_odd; // ((15 - 127) << 23) + 0xFFF, modulo 2³²
   return sign | static_cast<std::uint16_t>(bits >> 13U);
}

} // namespace rt
