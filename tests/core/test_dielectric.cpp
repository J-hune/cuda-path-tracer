#include "chi2_test.h"
#include "core/bsdf.h"
#include "core/dielectric.h"
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
constexpr float kGlassIor = 1.5f;
constexpr rt::Vec3 kUp{0, 1, 0}; // normale extérieure d'une surface horizontale
} // namespace

TEST_CASE("Fresnel : 4 % à incidence normale, 100 % en rasant", "[dielectric]") {
   CHECK_THAT(rt::fresnel_schlick(1.0f, kGlassIor), WithinAbs(0.04, 1e-6));
   CHECK_THAT(rt::fresnel_schlick(0.0f, kGlassIor), WithinAbs(1.0, 1e-6));
}

TEST_CASE("reflect : miroir autour de la normale", "[dielectric]") {
   STATIC_REQUIRE(rt::reflect({1, -1, 0}, kUp) == rt::Vec3{1, 1, 0});
}

TEST_CASE("Beer-Lambert : exp(-σ d)", "[dielectric]") {
   const rt::Vec3 t = rt::beer_lambert({1.0f, 0.0f, 2.0f}, 0.5f);
   CHECK_THAT(t.x, WithinAbs(std::exp(-0.5), 1e-6));
   CHECK_THAT(t.y, WithinAbs(1.0, 1e-6));
   CHECK_THAT(t.z, WithinAbs(std::exp(-1.0), 1e-6));
}

TEST_CASE("réfraction : loi de Snell en entrant dans le verre", "[dielectric]") {
   // Incidence à 45° depuis l'air : sin(θt) = sin(45°) / 1,5
   const rt::Vec3 d = rt::normalize({1, -1, 0});
   rt::Pcg32 rng(1, 0);
   int transmitted = 0;
   for (int i = 0; i < 1000; ++i) {
      const rt::DielectricSample s = rt::sample_dielectric(d, kUp, kGlassIor, rng);
      REQUIRE_THAT(rt::length(s.direction), WithinAbs(1.0, 1e-5));
      if (s.transmitted) {
         ++transmitted;
         REQUIRE(s.direction.y < 0.0f); // passe sous la surface
         REQUIRE_THAT(s.direction.x, WithinAbs(std::sin(std::acos(-1.0) / 4) / kGlassIor, 1e-5));
      } else {
         REQUIRE(s.direction == rt::reflect(d, kUp));
      }
   }
   // À 45°, Fresnel réfléchit environ 5 % : la grande majorité est transmise
   CHECK(transmitted > 900);
}

TEST_CASE("réflexion totale interne en sortant du verre", "[dielectric]") {
   // Depuis l'intérieur, au-delà de l'angle critique (41,8° pour 1,5) : toujours réfléchi
   const rt::Vec3 d = rt::normalize({1, 0.5f, 0}); // 63° de la normale, vers l'extérieur
   rt::Pcg32 rng(2, 0);
   for (int i = 0; i < 100; ++i) {
      const rt::DielectricSample s = rt::sample_dielectric(d, kUp, kGlassIor, rng);
      REQUIRE_FALSE(s.transmitted);
      REQUIRE(s.direction.y < 0.0f); // reste dans le verre
   }
}

namespace {

constexpr int kSampleCount = 400'000;
constexpr int kChi2Samples = 1'000'000;
constexpr std::array<float, 3> kAlphas{0.05f, 0.3f, 0.8f};
// Depuis l'air vers le verre, puis depuis le verre vers l'air
constexpr std::array<float, 2> kEtas{kGlassIor, 1.0f / kGlassIor};

// Directions d'observation : verticale, moyenne, rasante
std::array<rt::Vec3, 3> outgoing_directions() {
   return {
      rt::Vec3{0.0f, 0.0f, 1.0f},
      rt::normalize({0.6f, 0.2f, 0.7f}),
      rt::normalize({0.95f, -0.1f, 0.15f})
   };
}

} // namespace

TEST_CASE("verre dépoli : sample, eval et pdf sont cohérents", "[dielectric]") {
   rt::Pcg32 rng(7, 0);
   for (const float alpha : kAlphas) {
      for (const float eta : kEtas) {
         for (const rt::Vec3 wo : outgoing_directions()) {
            for (int i = 0; i < 10'000; ++i) {
               const rt::BsdfSample s = rt::rough_dielectric::sample(
                  wo,
                  alpha,
                  eta,
                  {rng.next_float(), rng.next_float()},
                  rng.next_float()
               );
               if (s.pdf <= 0.0f) {
                  continue;
               }
               const float pdf = rt::rough_dielectric::pdf(wo, s.wi, alpha, eta);
               const float f = rt::rough_dielectric::eval(wo, s.wi, alpha, eta);
               REQUIRE_THAT(s.pdf, WithinAbs(pdf, 1e-4 * pdf));
               REQUIRE_THAT(s.weight.x, WithinAbs(f * std::fabs(s.wi.z) / pdf, 1e-4));
               // Seule la part masquée est perdue : G2 / G1 <= 1
               REQUIRE(s.weight.x <= 1.0f + 1e-4f);
            }
         }
      }
   }
}

TEST_CASE("verre dépoli : la densité s'intègre à la part des tirages réussis", "[dielectric]") {
   // Les tirages qui passent du mauvais côté de la surface moyenne manquent à l'intégrale
   rt::Pcg32 rng(7, 1);
   for (const float alpha : kAlphas) {
      for (const float eta : kEtas) {
         const rt::Vec3 wo = rt::normalize({0.6f, 0.2f, 0.7f});
         double integral = 0.0;
         int succeeded = 0;
         for (int i = 0; i < kSampleCount; ++i) {
            const rt::Vec3 wi = rt::random_unit_vector(rng);
            integral += rt::rough_dielectric::pdf(wo, wi, alpha, eta) * 4.0 * rt::kPi;
            const rt::BsdfSample s = rt::rough_dielectric::sample(
               wo,
               alpha,
               eta,
               {rng.next_float(), rng.next_float()},
               rng.next_float()
            );
            succeeded += s.pdf > 0.0f ? 1 : 0;
         }
         INFO("alpha = " << alpha << ", eta = " << eta);
         CHECK_THAT(integral / kSampleCount, WithinAbs(double(succeeded) / kSampleCount, 0.02));
      }
   }
}

TEST_CASE("verre dépoli : lobe étroit à incidence normale", "[dielectric]") {
   // Part des directions réfractées à moins de 1° de -z : comptée sur les tirages, et intégrée
   constexpr float kAlpha = 0.05f;
   constexpr double kCap = rt::kPi / 180.0;
   constexpr int kSteps = 4000;
   const rt::Vec3 wo{0.0f, 0.0f, 1.0f};
   for (const float eta : kEtas) {
      rt::Pcg32 rng(7, 3);
      int inside = 0;
      for (int i = 0; i < kSampleCount; ++i) {
         const rt::BsdfSample s = rt::rough_dielectric::sample(
            wo,
            kAlpha,
            eta,
            {rng.next_float(), rng.next_float()},
            rng.next_float()
         );
         inside += s.pdf > 0.0f && s.wi.z < -std::cos(kCap) ? 1 : 0;
      }
      double integral = 0.0;
      for (int i = 0; i < kSteps; ++i) {
         const double theta = (i + 0.5) * kCap / kSteps;
         const rt::Vec3 wi{
            static_cast<float>(std::sin(theta)),
            0.0f,
            static_cast<float>(-std::cos(theta))
         };
         integral += rt::rough_dielectric::pdf(wo, wi, kAlpha, eta) * std::sin(theta) *
                     (kCap / kSteps) * 2.0 * rt::kPi;
      }
      INFO("eta = " << eta);
      CHECK_THAT(double(inside) / kSampleCount, WithinAbs(integral, 0.005));
   }
}

TEST_CASE("verre dépoli : peu rugueux, il conserve presque toute l'énergie", "[dielectric]") {
   // ∫ f |cos θi| dω, estimé par la moyenne des poids tirés, hors incidence rasante (masquage)
   rt::Pcg32 rng(7, 2);
   for (const float eta : kEtas) {
      for (const rt::Vec3 wo : {outgoing_directions()[0], outgoing_directions()[1]}) {
         double sum = 0.0;
         for (int i = 0; i < kSampleCount; ++i) {
            sum += rt::rough_dielectric::sample(
                      wo,
                      0.05f,
                      eta,
                      {rng.next_float(), rng.next_float()},
                      rng.next_float()
            )
                      .weight.x;
         }
         INFO("eta = " << eta << ", wo.z = " << wo.z);
         CHECK(sum / kSampleCount > 0.97);
         CHECK(sum / kSampleCount <= 1.0);
      }
   }
}

TEST_CASE("verre dépoli : khi-deux entre sample et pdf", "[dielectric][chi2]") {
   // Seuil de rejet corrigé (Bonferroni) pour les 18 tests. Le lobe le plus étroit est plus fin
   // que la quadrature des cases : testé à part
   constexpr double kSignificance = 0.01 / 18.0;
   for (const float alpha : {0.1f, 0.3f, 0.8f}) {
      for (const float eta : kEtas) {
         for (const rt::Vec3 wo : outgoing_directions()) {
            rt::Pcg32 lobe_rng(9, 0);
            const double p = chi2::p_value(
               [&](rt::Vec2 u) {
                  const rt::BsdfSample
                     s = rt::rough_dielectric::sample(wo, alpha, eta, u, lobe_rng.next_float());
                  return s.pdf > 0.0f ? s.wi : rt::Vec3{0, 0, 0};
               },
               [&](rt::Vec3 wi) { return rt::rough_dielectric::pdf(wo, wi, alpha, eta); },
               kChi2Samples,
               rt::Pcg32(42, 0)
            );
            INFO("alpha = " << alpha << ", eta = " << eta << ", wo.z = " << wo.z);
            INFO("p-valeur = " << p);
            CHECK(p > kSignificance);
         }
      }
   }
}
