//
// Created by jay on 11/12/24.
//

#ifndef FPS_COUNTER_CUH
#define FPS_COUNTER_CUH

#include "shader.cuh"

namespace cu_sim {
class fps_counter {
public:
  fps_counter(const fps_counter &) = delete;
  fps_counter(fps_counter &&) = delete;

  fps_counter &operator=(const fps_counter &) = delete;
  fps_counter &operator=(fps_counter &&) = delete;

  static fps_counter &get();
  void draw() const;

  ~fps_counter();
private:
  fps_counter();

  unsigned int vbo = 0;
  unsigned int vao = 0;
  unsigned int ebo = 0;
  unsigned int tex_id = 0;
  shader s;
  float aspect = 2.0f; // h / w
};
}

#endif //FPS_COUNTER_CUH
