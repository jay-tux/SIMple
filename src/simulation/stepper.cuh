//
// Created by jay on 9/30/24.
//

#ifndef STEPPER_CUH
#define STEPPER_CUH

#include <vector>

#include "rendering/object.cuh"
#include "rendering/line.cuh"
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
  explicit stepper(const buffer_handles &buffers, const line_handles &handles, const std::vector<body> &initial_state, size_t hist_size, size_t hist_skip);
  stepper(const stepper &) = delete;
  stepper(stepper &&) = delete;
  stepper &operator=(const stepper &) = delete;
  stepper &operator=(stepper &&) = delete;

  void step(float dt, size_t frame_idx);

  ~stepper();
private:
  cuda_gl_bridge<float3> pos_buf;
  cuda_gl_bridge<float> radius_buf;
  cuda_gl_bridge<float3> color_buf;
  cuda_gl_bridge<float3> history_buf;
  cuda_gl_bridge<float3> line_color_buf;
  body *bodies = nullptr;
  body *back_buffer = nullptr;
  dim3 grid;
  dim3 block;
  size_t history_length;
  size_t history_skip;
};
}

#endif //STEPPER_CUH
