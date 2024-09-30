//
// Created by jay on 9/30/24.
//

#ifndef DATA_CUH
#define DATA_CUH

namespace cu_sim {
struct vec {
  float x;
  float y;
  float z;

  __host__ __device__ vec operator+(const vec &other) const {
    return {x + other.x, y + other.y, z + other.z};
  }
  __host__ __device__ vec operator-(const vec &other) const {
    return {x - other.x, y - other.y, z - other.z};
  }
  __host__ __device__ vec operator*(const float scalar) const {
    return {x * scalar, y * scalar, z * scalar};
  }
  __host__ __device__ friend vec operator*(const float scalar, const vec &v) {
    return v * scalar;
  }
  __host__ __device__ vec operator/(const float scalar) const {
    return {x / scalar, y / scalar, z / scalar};
  }
  __host__ __device__ vec &operator+=(const vec &other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
  }
  __host__ __device__ vec &operator-=(const vec &other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
  }
  __host__ __device__ vec &operator*=(const float scalar) {
    x *= scalar;
    y *= scalar;
    z *= scalar;
    return *this;
  }
  __host__ __device__ vec &operator/=(const float scalar) {
    x /= scalar;
    y /= scalar;
    z /= scalar;
    return *this;
  }
  __host__ __device__ vec operator-() const {
    return {-x, -y, -z};
  }
  __host__ __device__ float dot(const vec &other) const {
    return x * other.x + y * other.y + z * other.z;
  }
  __host__ __device__ vec cross(const vec &other) const {
    return {y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x};
  }
  __host__ __device__ float length() const {
    return sqrtf(x * x + y * y + z * z);
  }
  __host__ __device__ vec normalized() const {
    return *this / length();
  }
  __host__ __device__ vec nan2zero() const {
    return {isnan(x) ? 0 : x, isnan(y) ? 0 : y, isnan(z) ? 0 : z};
  }
};
}

#endif //DATA_CUH
