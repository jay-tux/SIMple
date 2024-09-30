//
// Created by jay on 9/30/24.
//

#ifndef CAMERA_CUH
#define CAMERA_CUH

#include <glm/glm.hpp>

namespace cu_sim {
class camera {
public:
  constexpr camera(const glm::vec3 &eye, const glm::vec3 &look_at, const glm::vec3 &up)
      : eye{eye}, look_at{look_at}, up{up} {}

  glm::mat4 view_matrix() const;
  glm::mat4 projection_matrix() const;

  glm::vec3 eye;
  glm::vec3 look_at;
  glm::vec3 up;
  float fov_y = 45.0f;
};
}

#endif //CAMERA_CUH
