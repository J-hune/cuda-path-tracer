#pragma once
#include "core/host_device.h"
#include "core/vec3.h"

namespace rt {

// Lumière venant de l'infini, là où le rayon ne touche rien.
// Dégradé vertical de bottom (vers le bas) à top (vers le haut), uniforme si les deux sont égaux.
// Noir par défaut : glTF ne décrit pas d'environnement.
struct Environment {
   Vec3 bottom{0.0f, 0.0f, 0.0f};
   Vec3 top{0.0f, 0.0f, 0.0f};

   [[nodiscard]] RT_HOST_DEVICE constexpr Vec3 radiance(Vec3 direction) const {
      const float t = 0.5f * (direction.y + 1.f);
      return ((1.f - t) * bottom) + (t * top);
   }
};

} // namespace rt
