#pragma once

#include "core/host_device.h"
#include "core/vec3.h"

#include <cmath>

namespace rt {

// Longueurs d'onde visibles, en nanomètres
inline constexpr float kWavelengthMin = 380.0f;
inline constexpr float kWavelengthMax = 780.0f;

// Gaussienne asymétrique des approximations de Wyman, Sloan et Shirley
RT_HOST_DEVICE inline float piecewise_gaussian(
   float x,
   float mu,
   float sigma_low,
   float sigma_high
) {
   const float t = (x - mu) / (x < mu ? sigma_low : sigma_high);
   return std::exp(-0.5f * t * t);
}

// Couleur sRGB linéaire d'une longueur d'onde. Fonctions colorimétriques CIE 1931 de Wyman, Sloan
// et Shirley (« Simple Analytic Approximations to the CIE XYZ Color Matching Functions », 2013),
// passées en sRGB. Les composantes négatives (hors gamut) sont mises à zéro, puis chaque canal est
// normalisé : sa moyenne sur [380, 780] nm vaut 1, une lumière blanche reste blanche.
RT_HOST_DEVICE inline Vec3 wavelength_rgb(float lambda) {
   const float x = (1.056f * piecewise_gaussian(lambda, 599.8f, 37.9f, 31.0f)) +
                   (0.362f * piecewise_gaussian(lambda, 442.0f, 16.0f, 26.7f)) -
                   (0.065f * piecewise_gaussian(lambda, 501.1f, 20.4f, 26.2f));
   const float y = (0.821f * piecewise_gaussian(lambda, 568.8f, 46.9f, 40.5f)) +
                   (0.286f * piecewise_gaussian(lambda, 530.9f, 16.3f, 31.1f));
   const float z = (1.217f * piecewise_gaussian(lambda, 437.0f, 11.8f, 36.0f)) +
                   (0.681f * piecewise_gaussian(lambda, 459.0f, 26.0f, 13.8f));
   const Vec3 rgb{
      (3.2406f * x) - (1.5372f * y) - (0.4986f * z),
      (-0.9689f * x) + (1.8758f * y) + (0.0415f * z),
      (0.0557f * x) - (0.2040f * y) + (1.0570f * z),
   };
   const Vec3 normalization{2.270371f, 3.466611f, 3.659781f};
   return max(rgb, Vec3{0.0f, 0.0f, 0.0f}) * normalization;
}

RT_HOST_DEVICE inline float sample_wavelength(float u) {
   return kWavelengthMin + (u * (kWavelengthMax - kWavelengthMin));
}

// Indice de réfraction à la longueur d'onde lambda (KHR_materials_dispersion) : loi de Cauchy à
// partir de l'indice à 587,6 nm et du nombre d'Abbe, qui vaut 20 / dispersion
RT_HOST_DEVICE inline float dispersive_ior(float ior, float dispersion, float lambda) {
   if (dispersion <= 0.0f) {
      return ior;
   }
   const float abbe = 20.0f / dispersion;
   const float n = ior + ((ior - 1.0f) / abbe * ((523655.0f / (lambda * lambda)) - 1.5168f));
   return std::fmax(n, 1.0f);
}

} // namespace rt
