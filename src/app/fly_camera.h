#pragma once

#include "core/camera.h"
#include "core/vec3.h"

// Caméra libre (« fly camera ») : une position et deux angles, lacet (yaw) et tangage (pitch).
// Le haut est toujours +y : la caméra ne peut pas basculer sur le côté.
class FlyCamera {
public:
   explicit FlyCamera(const rt::CameraSettings& initial);

   // Déplacement dans le repère de la caméra : x vers la droite, y vers le haut, z vers l'avant
   void move(rt::Vec3 local_offset);

   // Rotation, en radians
   void rotate(float delta_yaw, float delta_pitch);

   [[nodiscard]] rt::CameraSettings settings() const;

private:
   [[nodiscard]] rt::Vec3 forward() const;

   rt::Vec3 position_;
   float yaw_ = 0.0f;
   float pitch_ = 0.0f;
   float vertical_fov_degrees_;
};
