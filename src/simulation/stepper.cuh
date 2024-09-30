//
// Created by jay on 9/30/24.
//

#ifndef STEPPER_CUH
#define STEPPER_CUH

#include <vector>

#include "rendering/object.cuh"
#include "cuda_gl_bridge.cuh"
#include "data.cuh"

namespace cu_sim {
constexpr static float G = 6.67430e-11f;

struct body {
  vec position;
  vec velocity;
  float mass;
  float radius;
  vec color;
};

class stepper {
public:
  explicit stepper(const buffer_handles &buffers, const std::vector<body> &initial_state);
  stepper(const stepper &) = delete;
  stepper(stepper &&) = delete;
  stepper &operator=(const stepper &) = delete;
  stepper &operator=(stepper &&) = delete;

  void step(float dt);

  ~stepper();
private:
  cuda_gl_bridge<float3> pos_buf;
  cuda_gl_bridge<float> radius_buf;
  cuda_gl_bridge<float3> color_buf;
  body *bodies;
  body *back_buffer;
  dim3 grid;
  dim3 block;
};
}

#endif //STEPPER_CUH
