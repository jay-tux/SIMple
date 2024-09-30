//
// Created by jay on 9/30/24.
//

#include <stdexcept>

#include "stepper.cuh"

using namespace cu_sim;

stepper::stepper(const buffer_handles &buffers, const std::vector<body> &initial_state)
  : pos_buf{buffers.pos}, radius_buf{buffers.radius}, color_buf{buffers.color}
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

  cudaDeviceProp prop;
  cuda_checked(cudaGetDeviceProperties(&prop, 0));
  grid = dim3{static_cast<unsigned int>(count / prop.maxThreadsPerBlock + 1), 1, 1};
  block = dim3{static_cast<unsigned int>(prop.maxThreadsPerBlock), 1, 1};
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

  // printf(
  //   "[Thread %05u] - (%8f, %8f, %8f) - %8f - (%8f, %8f, %8f)\n"
  //   "              - velocity: (%8f, %8f, %8f)\n",
  //   idx,
  //   pos[idx].x, pos[idx].y, pos[idx].z,
  //   radius[idx],
  //   color[idx].x, color[idx].y, color[idx].z,
  //   bodies[idx].velocity.x, bodies[idx].velocity.y, bodies[idx].velocity.z
  // );
}

void stepper::step(const float dt) {
  step_kernel<<<grid, block>>>(bodies, back_buffer, pos_buf.element_count(), dt);
  cuda_checked(cudaPeekAtLastError());
  cuda_checked(cudaDeviceSynchronize());
  // implicit synchronization - forces all threads to complete before copying
  copy_kernel<<<grid, block>>>(bodies, back_buffer, pos_buf, radius_buf, color_buf, pos_buf.element_count());
  cuda_checked(cudaPeekAtLastError());
  cuda_checked(cudaDeviceSynchronize());
}

stepper::~stepper() {
  cudaFree(bodies);
}