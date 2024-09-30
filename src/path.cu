//
// Created by jay on 9/30/24.
//

#include <whereami.h>

#include "path.cuh"

std::string bin_path() {
  const auto len = wai_getExecutablePath(nullptr, 0, nullptr);
  std::string path(len, '\0');
  wai_getExecutablePath(path.data(), len, nullptr);
  return path;
}

const std::filesystem::path &cu_sim::binary_path() {
  static auto path = absolute(std::filesystem::path(bin_path())).parent_path();
  return path;
}
