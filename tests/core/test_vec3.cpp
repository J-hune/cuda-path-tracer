#include "core/vec3.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace {
void check_near(rt::Vec3 a, rt::Vec3 b, float epsilon = 1e-6f) {
   CHECK_THAT(a.x, Catch::Matchers::WithinAbs(b.x, epsilon));
   CHECK_THAT(a.y, Catch::Matchers::WithinAbs(b.y, epsilon));
   CHECK_THAT(a.z, Catch::Matchers::WithinAbs(b.z, epsilon));
}
} // namespace

TEST_CASE("égalité et différence de deux vecteurs", "[vec3]") {
   STATIC_REQUIRE(rt::Vec3{1, 2, 3} == rt::Vec3{1, 2, 3});
   STATIC_REQUIRE(rt::Vec3{1, 2, 3} != rt::Vec3{1, 2, 4});
   STATIC_REQUIRE(rt::Vec3{1, 2, 3} != rt::Vec3{1, 5, 3});
   STATIC_REQUIRE(rt::Vec3{1, 2, 3} != rt::Vec3{0, 2, 3});
}

TEST_CASE("addition de deux vecteurs", "[vec3]") {
   const rt::Vec3 a{1, 2, 3};
   const rt::Vec3 b{4, 5, 6};

   const rt::Vec3 sum = a + b;
   check_near(sum, {5, 7, 9});
}

TEST_CASE("soustraction de deux vecteurs", "[vec3]") {
   const rt::Vec3 a{1, 2, 3};
   const rt::Vec3 b{4, 5, 6};

   const rt::Vec3 diff = a - b;
   check_near(diff, {-3, -3, -3});
}

TEST_CASE("négation d'un vecteur", "[vec3]") {
   const rt::Vec3 v{1, -2, 3};

   const rt::Vec3 neg = -v;
   check_near(neg, {-1, 2, -3});
}

TEST_CASE("multiplication d'un vecteur par un scalaire", "[vec3]") {
   const rt::Vec3 v{1, 2, 3};
   const float s = 2.0f;

   const rt::Vec3 prod = v * s;
   check_near(prod, {2, 4, 6});
}

TEST_CASE("multiplication d'un scalaire par un vecteur", "[vec3]") {
   const rt::Vec3 v{1, 2, 3};
   const float s = 2.0f;

   const rt::Vec3 prod = s * v;
   check_near(prod, {2, 4, 6});
}

TEST_CASE("division d'un vecteur par un scalaire", "[vec3]") {
   const rt::Vec3 v{2, 4, 6};
   const float s = 2.0f;

   const rt::Vec3 div = v / s;
   check_near(div, {1, 2, 3});
}

TEST_CASE("multiplication de deux vecteurs", "[vec3]") {
   const rt::Vec3 a{1, 2, 3};
   const rt::Vec3 b{4, 5, 6};

   const rt::Vec3 prod = a * b;
   check_near(prod, {4, 10, 18});
}

TEST_CASE("addition d'un vecteur à un autre", "[vec3]") {
   rt::Vec3 v{1, 2, 3};
   v += {4, 5, 6};
   check_near(v, {5, 7, 9});
}

TEST_CASE("soustraction d'un vecteur à un autre", "[vec3]") {
   rt::Vec3 v{1, 2, 3};
   v -= {4, 5, 6};
   check_near(v, {-3, -3, -3});
}

TEST_CASE("multiplication d'un vecteur par un scalaire (in place)", "[vec3]") {
   rt::Vec3 v{1, 2, 3};
   v *= 2.0f;
   check_near(v, {2, 4, 6});
}

TEST_CASE("division d'un vecteur par un scalaire (in place)", "[vec3]") {
   rt::Vec3 v{2, 4, 6};
   v /= 2.0f;
   check_near(v, {1, 2, 3});
}

TEST_CASE("produit scalaire de deux vecteurs", "[vec3]") {
   const rt::Vec3 a{1, 2, 3};
   const rt::Vec3 b{4, 5, 6};

   const float dot_product = rt::dot(a, b);
   CHECK(dot_product == 32.0f);

   CHECK(rt::length_squared(a) == 14.0f);

   // Deux vecteurs perpendiculaires ont un produit scalaire nul
   CHECK(rt::dot(rt::Vec3{1, 0, 0}, rt::Vec3{0, 1, 0}) == 0.0f);
}

TEST_CASE("évaluation à la compilation des fonctions constexpr", "[vec3]") {
   constexpr rt::Vec3 kAxisX{1, 0, 0};
   constexpr rt::Vec3 kAxisY{0, 1, 0};

   STATIC_REQUIRE(rt::dot(rt::Vec3{1, 2, 3}, rt::Vec3{4, 5, 6}) == 32.0f);
   STATIC_REQUIRE(rt::length_squared(rt::Vec3{1, 2, 3}) == 14.0f);
   STATIC_REQUIRE(rt::cross(kAxisX, kAxisY).x == 0.0f);
   STATIC_REQUIRE(rt::cross(kAxisX, kAxisY).y == 0.0f);
   STATIC_REQUIRE(rt::cross(kAxisX, kAxisY).z == 1.0f);
   STATIC_REQUIRE((kAxisX + kAxisY).x == 1.0f);
   STATIC_REQUIRE((kAxisX - kAxisY).y == -1.0f);
}

TEST_CASE("produit vectoriel de deux vecteurs", "[vec3]") {
   const rt::Vec3 a{1, 0, 0};
   const rt::Vec3 b{0, 1, 0};

   const rt::Vec3 cross_product = rt::cross(a, b);
   check_near(cross_product, {0, 0, 1});

   const rt::Vec3 cross_product_reverse = rt::cross(b, a);
   check_near(cross_product_reverse, {0, 0, -1});

   CHECK(rt::dot(cross_product, a) == 0.0f);
   CHECK(rt::dot(cross_product, b) == 0.0f);
}

TEST_CASE("longueur d'un vecteur", "[vec3]") {
   const rt::Vec3 v{3, 0, 4};

   const float norm = rt::length(v);
   CHECK_THAT(norm, Catch::Matchers::WithinAbs(5.0f, 1e-6f));
}

TEST_CASE("normalisation d'un vecteur", "[vec3]") {
   const rt::Vec3 v{3, 0, 4};

   const rt::Vec3 normalized = rt::normalize(v);
   check_near(normalized, {0.6f, 0.0f, 0.8f});

   const float norm = rt::length(normalized);
   CHECK_THAT(norm, Catch::Matchers::WithinAbs(1.0f, 1e-6f));
}
