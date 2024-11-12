//
// Created by jay on 9/30/24.
//

#ifndef CUDA_GL_BRIDGE_CUH
#define CUDA_GL_BRIDGE_CUH

#include <cuda_gl_bridge.cuh>
#include <stdexcept>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <vector>
#include <format>

namespace cu_sim {
inline void _cuda_checked(const cudaError_t err, const char *call, const char *file, const int line) {
  if(err != cudaSuccess) {
    throw std::runtime_error(std::format("{}:{}: {} (during {})", file, line, cudaGetErrorString(err), call));
  }
}
#define cuda_checked(X) _cuda_checked(X, #X, __FILE__, __LINE__)

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

template <typename T>
class cuda_gl_bridge_set {
public:
  template <typename F>
  explicit cuda_gl_bridge_set(const std::vector<unsigned int> &handles, F &&f) {
    cuda_checked(cudaSetDevice(0));
    for(const auto &handle: handles) {
      cudaGraphicsResource *res;
      T *dev_ptr;
      size_t size;
      cuda_checked(cudaGraphicsGLRegisterBuffer(&res, handle, cudaGraphicsRegisterFlagsNone));
      cuda_checked(cudaGraphicsMapResources(1, &res));
      cuda_checked(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void **>(&dev_ptr), &size, res));

      resources.emplace_back(res);
      dev_ptrs_host.emplace_back(dev_ptr);
      sizes.emplace_back(size);
      elem_counts.emplace_back(size / sizeof(T));
      f(handle, size, size / sizeof(T));
    }

    cuda_checked(cudaMallocManaged(&dev_ptrs, handles.size() * sizeof(T *)));
    cuda_checked(cudaMemcpy(dev_ptrs, dev_ptrs_host.data(), handles.size() * sizeof(T *), cudaMemcpyHostToDevice));
  }

  cuda_gl_bridge_set(const cuda_gl_bridge_set &) = delete;
  cuda_gl_bridge_set &operator=(const cuda_gl_bridge_set &) = delete;

  // ReSharper disable once CppNonExplicitConversionOperator
  constexpr operator T **() { return dev_ptrs; }
  // ReSharper disable once CppNonExplicitConversionOperator
  constexpr operator const T **() const { return dev_ptrs; }
  constexpr std::vector<size_t> byte_sizes() const { return sizes; }
  constexpr std::vector<size_t> element_counts() const { return elem_counts; }

  __device__ constexpr T *operator[](const size_t idx) { return dev_ptrs[idx]; }
  __device__ constexpr const T *operator[](const size_t idx) const { return dev_ptrs[idx]; }

  ~cuda_gl_bridge_set() {
    for(size_t i = 0; i < resources.size(); i++) {
      cudaGraphicsUnmapResources(1, &resources[i]);
      cudaGraphicsUnregisterResource(resources[i]);
    }
      cudaFree(dev_ptrs);
    // cudaGraphicsUnmapResources(1, &resource);
    // cudaGraphicsUnregisterResource(resource);
  }
private:
  std::vector<cudaGraphicsResource *> resources;
  std::vector<T *> dev_ptrs_host;
  T **dev_ptrs;
  std::vector<size_t> sizes{};
  std::vector<size_t> elem_counts{};
};
}

#endif //CUDA_GL_BRIDGE_CUH
