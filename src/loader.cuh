//
// Created by jay on 9/30/24.
//

#ifndef LOADER_CUH
#define LOADER_CUH

#include <string>
#include <vector>

#include "simulation/stepper.cuh"

namespace cu_sim {
std::vector<body> load_bodies(const std::string &infile);
}

#endif //LOADER_CUH
