//
// Created by jay on 9/30/24.
//

#ifndef CAMERA_CUH
#define CAMERA_CUH

#include <glm/glm.hpp>
#include <random>

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

struct moving_cam {
  explicit moving_cam(const camera &cam, const float speed) : camera{cam}, p0{cam.eye}, p1{random() * 2.0f}, p2{random()}, t{0.0f}, speed{speed} {}

  camera camera;
  glm::vec3 p0;
  glm::vec3 p1;
  glm::vec3 p2;
  float t;
  float speed;

  void interpolate(const float dt) {
    t += speed * dt;
    if(t >= 1.0f) {
      t -= 1.0f;
      p0 = p2;
      p1 = 1.5f * random();
      p2 = random();
    }

    camera.eye = p1 + (1 - t) * (1 - t) * (p0 - p1) + t * t * (p2 - p1);
  }

  glm::vec3 random() const {
    static std::mt19937 rng{std::random_device{}()};
    static std::uniform_real_distribution<float> dist{-10.0f, 10.0f};
    const static float r = length(camera.eye - camera.look_at);

    return normalize(glm::vec3{dist(rng), dist(rng), dist(rng)}) * r;
  }
};
}

#endif //CAMERA_CUH
