//
// Created by jay on 9/30/24.
//

#ifndef LINE_CUH
#define LINE_CUH

#include "shader.cuh"

namespace cu_sim {
struct line_handles {
  unsigned int history;
  unsigned int color;
};

class hist_line {
public:
  hist_line(size_t history_size, size_t instances);
  hist_line(const hist_line &) = delete;
  hist_line &operator=(const hist_line &) = delete;

  void draw(const shader &s) const;
  constexpr const line_handles &cuda_handles() const { return cuda; }

  ~hist_line();

private:
  unsigned int vao = 0;
  unsigned int vbo = 0;
  unsigned int tbo = 0;
  line_handles cuda{};
  size_t history_size;
  size_t instance_count;
};
}

#endif //LINE_CUH
