#include "chi2_test.h"
#include "core/bsdf.h"
#include "core/ggx.h"
#include "core/rng.h"
#include "core/sampling.h"
#include "core/vec2.h"
#include "core/vec3.h"

#include <array>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

using Catch::Matchers::WithinAbs;

namespace {

constexpr int kSampleCount = 400'000;
constexpr int kChi2Samples = 1'000'000;
constexpr std::array<float, 3> kAlphas{0.05f, 0.3f, 0.8f};
constexpr std::array<float, 3> kRoughnesses{0.15f, 0.5f, 1.0f};

// Directions d'observation : verticale, moyenne, rasante
std::array<rt::Vec3, 3> outgoing_directions() {
   return {
      rt::Vec3{0.0f, 0.0f, 1.0f},
      rt::normalize({0.6f, 0.2f, 0.7f}),
      rt::normalize({0.95f, -0.1f, 0.15f})
   };
}

} // namespace

TEST_CASE("GGX : la distribution des normales est normalisée", "[ggx]") {
   // ∫ D(h) cos θh dωh = 1. D ne dépend que de θh : intégrale 1D par la méthode du point milieu
   constexpr int kSteps = 200'000;
   for (const float alpha : kAlphas) {
      double sum = 0.0;
      for (int i = 0; i < kSteps; ++i) {
         const double theta = (i + 0.5) * (rt::kPi / 2.0) / kSteps;
         const rt::Vec3 h{
            static_cast<float>(std::sin(theta)),
            0.0f,
            static_cast<float>(std::cos(theta))
         };
         sum += rt::ggx::distribution(h, alpha) * std::cos(theta) * std::sin(theta) * 2.0 *
                rt::kPi * (rt::kPi / 2.0 / kSteps);
      }
      INFO("alpha = " << alpha);
      CHECK_THAT(sum, WithinAbs(1.0, 1e-3));
   }
}

TEST_CASE("GGX : la distribution des normales visibles est normalisée", "[ggx]") {
   // ∫ G1(wo) max(0, wo·h) D(h) / wo.z dωh = 1 : vérifie le terme de masquage Λ.
   // Intégrale 2D (θh, φh) par la méthode du point milieu
   constexpr int kThetaSteps = 4000;
   constexpr int kPhiSteps = 400;
   for (const float alpha : kAlphas) {
      for (const rt::Vec3 wo : outgoing_directions()) {
         double sum = 0.0;
         for (int i = 0; i < kThetaSteps; ++i) {
            const double theta = (i + 0.5) * (rt::kPi / 2.0) / kThetaSteps;
            for (int j = 0; j < kPhiSteps; ++j) {
               const double phi = (j + 0.5) * (2.0 * rt::kPi) / kPhiSteps;
               const rt::Vec3 h{
                  static_cast<float>(std::sin(theta) * std::cos(phi)),
                  static_cast<float>(std::sin(theta) * std::sin(phi)),
                  static_cast<float>(std::cos(theta))
               };
               sum += rt::ggx::smith_g1(wo, alpha) * std::fmax(rt::dot(wo, h), 0.0f) *
                      rt::ggx::distribution(h, alpha) / wo.z * std::sin(theta) *
                      (rt::kPi / 2.0 / kThetaSteps) * (2.0 * rt::kPi / kPhiSteps);
            }
         }
         INFO("alpha = " << alpha << ", wo.z = " << wo.z);
         CHECK_THAT(sum, WithinAbs(1.0, 2e-3));
      }
   }
}

TEST_CASE("BSDF principled : sample, eval et pdf sont cohérents", "[principled]") {
   for (const float metallic : {0.0f, 0.5f, 1.0f}) {
      for (const float roughness : kRoughnesses) {
         const rt::SurfaceParams params{
            .base_color = {0.9f, 0.6f, 0.3f},
            .metallic = metallic,
            .roughness = roughness
         };
         for (const rt::Vec3 wo : outgoing_directions()) {
            rt::Pcg32 rng(4, 0);
            for (int i = 0; i < 2000; ++i) {
               const rt::BsdfSample s = rt::principled::sample(
                  params,
                  wo,
                  {rng.next_float(), rng.next_float()}
               );
               if (s.pdf <= 0.0f) {
                  continue;
               }
               REQUIRE_THAT(rt::length(s.wi), WithinAbs(1.0, 1e-4));
               REQUIRE_THAT(s.pdf, WithinAbs(rt::principled::pdf(params, wo, s.wi), s.pdf * 1e-4));
               const rt::Vec3 expected = rt::principled::eval(params, wo, s.wi) * (s.wi.z / s.pdf);
               REQUIRE_THAT(s.weight.x, WithinAbs(expected.x, 1e-3));
            }
         }
      }
   }
}

TEST_CASE("BSDF principled : réciprocité de Helmholtz", "[principled]") {
   rt::Pcg32 rng(4, 1);
   const rt::SurfaceParams params{
      .base_color = {0.8f, 0.4f, 0.2f},
      .metallic = 0.3f,
      .roughness = 0.4f
   };
   for (int i = 0; i < 1000; ++i) {
      rt::Vec3 a = rt::random_unit_vector(rng);
      rt::Vec3 b = rt::random_unit_vector(rng);
      a.z = std::fabs(a.z);
      b.z = std::fabs(b.z);
      const rt::Vec3 ab = rt::principled::eval(params, a, b);
      const rt::Vec3 ba = rt::principled::eval(params, b, a);
      REQUIRE_THAT(ab.x, WithinAbs(ba.x, std::fmax(ab.x, 1.0f) * 1e-4));
      REQUIRE_THAT(ab.y, WithinAbs(ba.y, std::fmax(ab.y, 1.0f) * 1e-4));
   }
}

TEST_CASE("BSDF principled : l'albédo intégré ne dépasse jamais 1", "[principled]") {
   // E[poids] = ∫ f cos θi dωi : la part de lumière renvoyée, au plus 100 %
   for (const float metallic : {0.0f, 1.0f}) {
      for (const float roughness : kRoughnesses) {
         const rt::SurfaceParams params{
            .base_color = {1, 1, 1},
            .metallic = metallic,
            .roughness = roughness
         };
         for (const rt::Vec3 wo : outgoing_directions()) {
            rt::Pcg32 rng(4, 2);
            double sum = 0.0;
            for (int i = 0; i < kSampleCount; ++i) {
               sum += rt::principled::sample(params, wo, {rng.next_float(), rng.next_float()})
                         .weight.x;
            }
            const double albedo = sum / kSampleCount;
            INFO(
               "metallic = " << metallic << ", roughness = " << roughness << ", wo.z = " << wo.z
                             << ", albédo = " << albedo
            );
            CHECK(albedo <= 1.0 + 0.005);
            // Modèle à une seule réflexion : un métal très rugueux perd jusqu'à ~70 % de l'énergie
            // (sera corrigé par la compensation multi-rebond)
            CHECK(albedo > 0.25);
         }
      }
   }
}

TEST_CASE("BSDF principled : khi-deux entre sample et pdf", "[principled][chi2]") {
   // 2 × 3 × 3 = 18 tests : seuil corrigé (Bonferroni)
   constexpr double kSignificance = 0.01 / 18.0;
   for (const float metallic : {0.0f, 1.0f}) {
      for (const float roughness : kRoughnesses) {
         const rt::SurfaceParams params{
            .base_color = {0.9f, 0.6f, 0.3f},
            .metallic = metallic,
            .roughness = roughness
         };
         for (const rt::Vec3 wo : outgoing_directions()) {
            const double p = chi2::p_value(
               [&](rt::Vec2 u) {
                  const rt::BsdfSample s = rt::principled::sample(params, wo, u);
                  return s.pdf > 0.0f ? s.wi : rt::Vec3{0, 0, 0};
               },
               [&](rt::Vec3 wi) { return rt::principled::pdf(params, wo, wi); },
               kChi2Samples,
               rt::Pcg32(42, 2)
            );
            INFO(
               "metallic = " << metallic << ", roughness = " << roughness << ", wo.z = " << wo.z
                             << ", p-valeur = " << p
            );
            CHECK(p > kSignificance);
         }
      }
   }
}
