//
// Created by jay on 9/30/24.
//

#include <glm/gtc/matrix_transform.hpp>

#include "camera.cuh"
#include "gl_wrapper.cuh"

using namespace cu_sim;

glm::mat4 camera::view_matrix() const {
  return lookAt(eye, look_at, up);
}

glm::mat4 camera::projection_matrix() const {
  return glm::perspective(glm::radians(fov_y), gl_wrapper::get().aspect(), 0.1f, 100.0f);
}