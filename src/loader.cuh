//
// Created by jay on 9/30/24.
//

#ifndef LOADER_CUH
#define LOADER_CUH

#include <string>
#include <vector>
#include <glm/vec3.hpp>

#include "simulation/stepper.cuh"

namespace cu_sim {
struct settings {
  float time_scale = 1e-3f;
  glm::vec3 eye{0.0f, 0.0f, 7.0f};
  glm::vec3 focus{0.0f, 0.0f, 0.0f};
  int history_size = 100;
  int history_skip = 100;
};

std::pair<std::vector<body>, settings> load_bodies(const std::string &infile);
}

#endif //LOADER_CUH
