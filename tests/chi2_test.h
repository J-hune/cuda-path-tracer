#pragma once

#include "core/rng.h"
#include "core/vec2.h"
#include "core/vec3.h"

#include <cmath>
#include <cstddef>
#include <vector>

// Test d'adéquation du khi-deux entre une fonction d'échantillonnage et la densité annoncée.
//
// La sphère est découpée en kZBins × kPhiBins cases d'angle solide égal (z = cos θ uniforme).
// Observé : nombre de directions tirées par sample dans chaque case.
// Attendu : intégrale de pdf sur la case (quadrature du point milieu), multipliée par le nombre de
// tirages. Les cases dont l'attendu est trop faible sont regroupées, comme le veut le test.
//
// sample(u) -> direction unitaire, ou vecteur nul si le tirage échoue ; pdf(direction) -> densité.
// Renvoie la p-valeur : la probabilité d'observer un écart au moins aussi grand par hasard.
namespace chi2 {

inline constexpr int kZBins = 20;
inline constexpr int kPhiBins = 40;
inline constexpr int kQuadrature = 8; // points par côté de case pour intégrer la densité
inline constexpr double kMinExpected = 5.0;

namespace detail {

inline int bin_index(rt::Vec3 d) {
   const double phi = std::atan2(d.y, d.x) + rt::kPi; // [0, 2π]
   const auto z_bin = static_cast<int>((d.z + 1.0) * 0.5 * kZBins);
   const auto phi_bin = static_cast<int>(phi / (2.0 * rt::kPi) * kPhiBins);
   const int z = z_bin < 0 ? 0 : (z_bin >= kZBins ? kZBins - 1 : z_bin);
   const int p = phi_bin < 0 ? 0 : (phi_bin >= kPhiBins ? kPhiBins - 1 : phi_bin);
   return (z * kPhiBins) + p;
}

// Q(a, x) = Γ(a, x) / Γ(a), fonction gamma incomplète régularisée supérieure
// (série pour x < a + 1, fraction continue de Lentz sinon ; Numerical Recipes, § 6.2)
inline double gamma_q(double a, double x) {
   if (x <= 0.0) {
      return 1.0;
   }
   const double log_prefix = (a * std::log(x)) - x - std::lgamma(a);
   if (x < a + 1.0) {
      double term = 1.0 / a;
      double sum = term;
      for (int n = 1; n < 10'000; ++n) {
         term *= x / (a + n);
         sum += term;
         if (std::fabs(term) < std::fabs(sum) * 1e-15) {
            break;
         }
      }
      return 1.0 - (sum * std::exp(log_prefix));
   }
   constexpr double kTiny = 1e-300;
   double b = x + 1.0 - a;
   double c = 1.0 / kTiny;
   double d = 1.0 / b;
   double h = d;
   for (int n = 1; n < 10'000; ++n) {
      const double an = -n * (n - a);
      b += 2.0;
      d = (an * d) + b;
      d = std::fabs(d) < kTiny ? kTiny : d;
      c = b + (an / c);
      c = std::fabs(c) < kTiny ? kTiny : c;
      d = 1.0 / d;
      const double delta = d * c;
      h *= delta;
      if (std::fabs(delta - 1.0) < 1e-15) {
         break;
      }
   }
   return std::exp(log_prefix) * h;
}

} // namespace detail

template <typename Sample, typename Pdf>
double p_value(Sample sample, Pdf pdf, int sample_count, rt::Pcg32 rng) {
   constexpr int kBins = kZBins * kPhiBins;

   std::vector<double> observed(kBins, 0.0);
   for (int i = 0; i < sample_count; ++i) {
      const rt::Vec3 d = sample(rt::Vec2{rng.next_float(), rng.next_float()});
      if (rt::length_squared(d) > 0.0f) {
         observed[static_cast<std::size_t>(detail::bin_index(d))] += 1.0;
      }
   }

   std::vector<double> expected(kBins, 0.0);
   const double cell_solid_angle = 4.0 * rt::kPi /
                                   (static_cast<double>(kBins) * kQuadrature * kQuadrature);
   for (int zb = 0; zb < kZBins; ++zb) {
      for (int pb = 0; pb < kPhiBins; ++pb) {
         double integral = 0.0;
         for (int i = 0; i < kQuadrature; ++i) {
            for (int j = 0; j < kQuadrature; ++j) {
               const double z = -1.0 + (2.0 * (zb + ((i + 0.5) / kQuadrature)) / kZBins);
               const double phi = (2.0 * rt::kPi * (pb + ((j + 0.5) / kQuadrature)) / kPhiBins) -
                                  rt::kPi;
               const double r = std::sqrt(std::fmax(0.0, 1.0 - (z * z)));
               const rt::Vec3 d{
                  static_cast<float>(r * std::cos(phi)),
                  static_cast<float>(r * std::sin(phi)),
                  static_cast<float>(z)
               };
               integral += pdf(d) * cell_solid_angle;
            }
         }
         expected
            [(static_cast<std::size_t>(zb) * kPhiBins) +
             static_cast<std::size_t>(pb)] = integral * sample_count;
      }
   }

   // Regroupe les cases à faible attendu dans une seule case, puis calcule le khi-deux
   double chi2 = 0.0;
   int degrees_of_freedom = -1;
   double pooled_observed = 0.0;
   double pooled_expected = 0.0;
   for (int b = 0; b < kBins; ++b) {
      const auto i = static_cast<std::size_t>(b);
      if (expected[i] < kMinExpected) {
         pooled_observed += observed[i];
         pooled_expected += expected[i];
         continue;
      }
      chi2 += (observed[i] - expected[i]) * (observed[i] - expected[i]) / expected[i];
      ++degrees_of_freedom;
   }
   if (pooled_expected >= kMinExpected) {
      chi2 += (pooled_observed - pooled_expected) * (pooled_observed - pooled_expected) /
              pooled_expected;
      ++degrees_of_freedom;
   } else if (pooled_observed > 0.0 && pooled_expected <= 0.0) {
      return 0.0; // des directions tombent là où la densité annonce zéro
   }
   if (degrees_of_freedom < 1) {
      return 1.0;
   }
   return detail::gamma_q(degrees_of_freedom / 2.0, chi2 / 2.0);
}

} // namespace chi2
