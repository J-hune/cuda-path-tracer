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

   [[nodiscard]] rt::Vec3 position() const { return position_; }
   void set_position(rt::Vec3 position) { position_ = position; }

   // Lacet nul : la caméra regarde vers -z, positif : elle tourne vers +x. Tangage positif : elle
   // regarde vers le haut. En radians.
   [[nodiscard]] float yaw() const { return yaw_; }
   [[nodiscard]] float pitch() const { return pitch_; }
   void set_orientation(float yaw, float pitch);
   [[nodiscard]] static float max_pitch();

   void set_vertical_fov(float degrees) { vertical_fov_degrees_ = degrees; }

   // Profondeur de champ : rayon de la lentille (0 : tout est net) et distance de mise au point
   void set_lens(float aperture_radius, float focus_distance);

private:
   [[nodiscard]] rt::Vec3 forward() const;

   rt::Vec3 position_;
   float yaw_ = 0.0f;
   float pitch_ = 0.0f;
   float vertical_fov_degrees_;
   float aperture_radius_;
   float focus_distance_;
};
