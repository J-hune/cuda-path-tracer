#include "core/path_tracer.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinAbs;

TEST_CASE("heuristique de puissance : les deux poids d'une direction font 1", "[mis]") {
   for (const float a : {0.01f, 0.5f, 1.0f, 3.0f, 250.0f}) {
      for (const float b : {0.01f, 0.5f, 1.0f, 3.0f, 250.0f}) {
         CHECK_THAT(rt::power_heuristic(a, b) + rt::power_heuristic(b, a), WithinAbs(1.0, 1e-6));
      }
   }
}

TEST_CASE("heuristique de puissance : cas limites", "[mis]") {
   // Si l'autre stratégie ne peut pas produire la direction, celle-ci a tout le poids
   CHECK(rt::power_heuristic(2.0f, 0.0f) == 1.0f);
   CHECK(rt::power_heuristic(0.0f, 2.0f) == 0.0f);
   CHECK(rt::power_heuristic(0.0f, 0.0f) == 0.0f);
   // Densités égales : poids égaux
   CHECK_THAT(rt::power_heuristic(1.5f, 1.5f), WithinAbs(0.5, 1e-6));
}
