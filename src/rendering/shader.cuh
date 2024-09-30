//
// Created by jay on 9/30/24.
//

#ifndef SHADER_CUH
#define SHADER_CUH

#include <string>
#include <glm/glm.hpp>

namespace cu_sim {
class shader {
public:
  shader(const std::string &vertex_path, const std::string &fragment_path);

  shader(const shader &) = delete;
  shader &operator=(const shader &) = delete;

  void enable() const;

  void set_m4(uint name, const glm::mat4 &value) const;
  void set_float(uint name, float value) const;

  ~shader();
private:
  unsigned int vertex_shader;
  unsigned int fragment_shader;
  unsigned int shader_id;
};
}

#endif //SHADER_CUH
