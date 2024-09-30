//
// Created by jay on 9/30/24.
//

#ifndef OBJECT_CUH
#define OBJECT_CUH

#include <string>

namespace cu_sim {
struct buffer_handles {
  unsigned int pos;
  unsigned int radius;
  unsigned int color;
};

class object {
public:
  object(const std::string &path, size_t instances);
  object(const object &) = delete;
  object &operator=(const object &) = delete;

  void draw() const;
  constexpr const buffer_handles &cuda_buffers() const { return cuda; }

  ~object();

private:
  unsigned int vao;
  unsigned int vbo;
  buffer_handles cuda;
  unsigned int ebo;
  size_t element_count;
  size_t instance_count;
};
}

#endif //OBJECT_CUH
