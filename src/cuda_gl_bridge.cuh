//
// Created by jay on 9/30/24.
//

#ifndef CUDA_GL_BRIDGE_CUH
#define CUDA_GL_BRIDGE_CUH

#include <cuda_gl_bridge.cuh>
#include <stdexcept>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

namespace cu_sim {
inline void cuda_checked(const cudaError_t err) {
  if(err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

template <typename T>
class cuda_gl_bridge {
public:
  explicit cuda_gl_bridge(const unsigned int handle) {
    cuda_checked(cudaSetDevice(0));
    cuda_checked(cudaGraphicsGLRegisterBuffer(&resource, handle, cudaGraphicsRegisterFlagsNone));
    cuda_checked(cudaGraphicsMapResources(1, &resource));
    cuda_checked(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void **>(&dev_ptr), &size, resource));
  }

  cuda_gl_bridge(const cuda_gl_bridge &) = delete;
  cuda_gl_bridge &operator=(const cuda_gl_bridge &) = delete;

  // ReSharper disable once CppNonExplicitConversionOperator
  constexpr operator T *() { return dev_ptr; }
  // ReSharper disable once CppNonExplicitConversionOperator
  constexpr operator const T *() const { return dev_ptr; }
  constexpr size_t byte_size() const { return size; }
  constexpr size_t element_count() const { return size / sizeof(T); }

  __device__ constexpr T &operator[](const size_t idx) { return dev_ptr[idx]; }
  __device__ constexpr const T &operator[](const size_t idx) const { return dev_ptr[idx]; }

  ~cuda_gl_bridge() {
    cudaGraphicsUnmapResources(1, &resource);
    cudaGraphicsUnregisterResource(resource);
  }
private:
  cudaGraphicsResource *resource;
  T *dev_ptr;
  size_t size;
};
}

#endif //CUDA_GL_BRIDGE_CUH
