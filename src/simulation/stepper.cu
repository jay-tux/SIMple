//
// Created by jay on 9/30/24.
//

#include <stdexcept>

#include "stepper.cuh"

using namespace cu_sim;

__global__ void setup_kernel(
  const body *__restrict__ bodies, float3 *__restrict__ history, float3 *__restrict__ line_color,
  const size_t count, const size_t hist_size
) {
  const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx >= count) return;
  for(size_t  i = 0; i < hist_size; i++)
    history[hist_size * idx + i] = float3(bodies[idx].position.x, bodies[idx].position.y, bodies[idx].position.z);
  line_color[idx] = float3(bodies[idx].color.x, bodies[idx].color.y, bodies[idx].color.z);
}

#include <iostream>
stepper::stepper(const buffer_handles &buffers, const line_handles &handles, const std::vector<body> &initial_state, const size_t hist_size, const size_t hist_skip)
  : pos_buf{buffers.pos}, radius_buf{buffers.radius}, color_buf{buffers.color}, history_buf{handles.history}, line_color_buf{handles.color},
    history_length{hist_size}, history_skip{hist_skip}
{
  const size_t count = initial_state.size();
  if(pos_buf.element_count() != count) {
    throw std::runtime_error("Mismatched buffer sizes: " + std::to_string(count) + " != " + std::to_string(pos_buf.element_count()));
  }
  if(radius_buf.element_count() != count) {
    throw std::runtime_error("Mismatched buffer sizes: " + std::to_string(count) + " != " + std::to_string(radius_buf.element_count()));
  }
  if(color_buf.element_count() != count) {
    throw std::runtime_error("Mismatched buffer sizes: " + std::to_string(count) + " != " + std::to_string(color_buf.element_count()));
  }

  cuda_checked(cudaMalloc(&bodies, count * sizeof(body)));
  cuda_checked(cudaMemcpy(bodies, initial_state.data(), count * sizeof(body), cudaMemcpyHostToDevice));
  cuda_checked(cudaMalloc(&back_buffer, count * sizeof(body)));
  cuda_checked(cudaMemcpy(back_buffer, initial_state.data(), count * sizeof(body), cudaMemcpyHostToDevice));

  cudaDeviceProp prop{};
  cuda_checked(cudaGetDeviceProperties(&prop, 0));
  grid = dim3{static_cast<unsigned int>(count / prop.maxThreadsPerBlock + 1), 1, 1};
  block = dim3{static_cast<unsigned int>(prop.maxThreadsPerBlock), 1, 1};

  setup_kernel<<<grid, block>>>(bodies, history_buf, line_color_buf, count, history_length);
  cuda_checked(cudaPeekAtLastError());
  cuda_checked(cudaDeviceSynchronize());
}

__global__ void step_kernel(
  const body *__restrict__ bodies, body *__restrict__ back,
  const size_t count, const float dt
) {
  const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx >= count) return;

  vec force{0, 0, 0};
  for(size_t i = 0; i < count; ++i) {
    if(idx == i) continue;
    vec direction = bodies[i].position - bodies[idx].position;
    const float distance = max(direction.length(), 1e-3f);
    direction = direction.normalized();
    force += direction * G * bodies[i].mass / (distance * distance); // F = G * m1 * m2 / r^2
  }

  // a = F / m
  const auto acceleration = force / bodies[idx].mass;

  // v += a * dt
  back[idx].velocity = bodies[idx].velocity + acceleration * dt;

  // x += v * dt
  back[idx].position = bodies[idx].position + back[idx].velocity * dt;
}

__global__ void copy_kernel(
  body *__restrict__ bodies, const body *__restrict__ back,
  float3 *__restrict__ pos, float *__restrict__ radius, float3 *__restrict__ color,
  float3 *__restrict__ history, float3 *__restrict__ line_color,
  const size_t count, const size_t history_size, const bool step_history
) {
  const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx >= count) return;

  bodies[idx] = back[idx];

  pos[idx].x = bodies[idx].position.x;
  pos[idx].y = bodies[idx].position.y;
  pos[idx].z = bodies[idx].position.z;

  radius[idx] = bodies[idx].radius;

  color[idx].x = bodies[idx].color.x;
  color[idx].y = bodies[idx].color.y;
  color[idx].z = bodies[idx].color.z;

  line_color[idx] = color[idx];
  if(step_history) {
    for(size_t i = history_size - 1; i > 0; i--) {
      history[idx * history_size + i] = history[idx * history_size + i - 1];
    }
  }
  history[idx * history_size] = pos[idx];
}

__global__ void copy_kernel_no_history(
  body *__restrict__ bodies, const body *__restrict__ back,
  float3 *__restrict__ pos, float *__restrict__ radius, float3 *__restrict__ color,
  const size_t count
) {
  const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx >= count) return;

  bodies[idx] = back[idx];

  pos[idx].x = bodies[idx].position.x;
  pos[idx].y = bodies[idx].position.y;
  pos[idx].z = bodies[idx].position.z;

  radius[idx] = bodies[idx].radius;

  color[idx].x = bodies[idx].color.x;
  color[idx].y = bodies[idx].color.y;
  color[idx].z = bodies[idx].color.z;
}

void stepper::step(const float dt, const size_t frame_idx) {
  step_kernel<<<grid, block>>>(bodies, back_buffer, pos_buf.element_count(), dt);
  cuda_checked(cudaPeekAtLastError());
  cuda_checked(cudaDeviceSynchronize());
  // explicit synchronization - forces all threads to complete before copying
  copy_kernel<<<grid, block>>>(
    bodies, back_buffer, pos_buf, radius_buf, color_buf, history_buf, line_color_buf, pos_buf.element_count(),
    history_length, (frame_idx % history_skip) == 0
  );
  cuda_checked(cudaPeekAtLastError());
  cuda_checked(cudaDeviceSynchronize());
}

void stepper::step_no_history(const float dt, const size_t frame_idx) {
  step_kernel<<<grid, block>>>(bodies, back_buffer, pos_buf.element_count(), dt);
  cuda_checked(cudaPeekAtLastError());
  cuda_checked(cudaDeviceSynchronize());
  // explicit synchronization - forces all threads to complete before copying
  copy_kernel_no_history<<<grid, block>>>(
    bodies, back_buffer, pos_buf, radius_buf, color_buf, pos_buf.element_count()
  );
  cuda_checked(cudaPeekAtLastError());
  cuda_checked(cudaDeviceSynchronize());
}

stepper::~stepper() {
  cudaFree(bodies);
}