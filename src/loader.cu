//
// Created by jay on 9/30/24.
//

#include <fstream>
#include <sstream>
#include <inipp.h>

#include "loader.cuh"

template <typename T, typename F>
requires(std::invocable<F, const std::string &> && std::convertible_to<std::invoke_result_t<F, const std::string &>, T>)
T get(const inipp::Ini<char>::Section &s, const std::string &section_name, const std::string &key, F &&convert) {
  if(const auto it = s.find(key); it != s.end()) {
    try {
      return convert(it->second);
    }
    catch(std::exception &ex) {
      throw std::runtime_error("Failed to convert value for key: " + key + " in section '" + section_name + "': " + ex.what());
    }
  }

  throw std::runtime_error("Missing key: " + key + " in section '" + section_name + "'");
}

template <typename T, typename F>
requires(std::invocable<F, const std::string &> && std::convertible_to<std::invoke_result_t<F, const std::string &>, T>)
T get(const inipp::Ini<char>::Section &s, const std::string &section_name, const std::string &key, const T &alt, F &&convert) {
  if(const auto it = s.find(key); it != s.end()) {
    try {
      return convert(it->second);
    }
    catch(std::exception &ex) {
      std::cerr << "Failed to convert value for key: " << key << " in section '" << section_name << "': " << ex.what() << "\n";
      return alt;
    }
  }

  return alt;
}

cu_sim::vec parse_vec(const std::string &str) {
  std::stringstream strm(str);
  cu_sim::vec res{};
  std::string buf;
  strm >> buf;
  res.x = std::stof(buf);
  strm >> buf;
  res.y = std::stof(buf);
  strm >> buf;
  res.z = std::stof(buf);
  return res;
}

bool true_false_on_off(const std::string &x) {
  if(x == "true" || x == "on") return true;
  if(x == "false" || x == "off") return false;
  throw std::runtime_error("Invalid value for mass_div_g: " + x);
}

std::pair<std::vector<cu_sim::body>, cu_sim::settings> cu_sim::load_bodies(const std::string &infile) {
  std::vector<body> res;
  settings set{};

  std::ifstream strm(infile);
  if(!strm.is_open()) {
    throw std::runtime_error("Failed to open file: " + infile);
  }

  inipp::Ini<char> parser;
  parser.parse(strm);

  res.reserve(parser.sections.size());

  for(const auto &[section, def]: parser.sections) {
    if(section == "SETTINGS") {
      set.time_scale = get<float>(def, section, "time_scale", set.time_scale, [](const std::string &x){ return std::stof(x); });
      set.eye = get<glm::vec3>(def, section, "eye", set.eye, parse_vec);
      set.focus = get<glm::vec3>(def, section, "focus", set.focus, parse_vec);
      continue;
    }
    body b{};

    b.position = get<vec>(def, section, "position", parse_vec);
    b.velocity = get<vec>(def, section, "velocity", parse_vec);
    b.mass = get<float>(def, section, "mass", [](const std::string &x) { return std::stof(x); });
    b.radius = get(def, section, "radius", 0.1f, [](const std::string &x) { return std::stof(x); });
    b.color = get(def, section, "color", vec{ 0.22f, 0.22f, 0.22f }, parse_vec);

    const bool mass_div_g = get(def, section, "mass_div_g", false, true_false_on_off);
    const bool pos_div_g = get(def, section, "pos_div_g", false, true_false_on_off);
    const bool vel_div_g = get(def, section, "vel_div_g", false, true_false_on_off);
    const bool radius_div_g = get(def, section, "radius_div_g", false, true_false_on_off);

    if(mass_div_g) b.mass /= G;
    if(pos_div_g) b.position /= G;
    if(vel_div_g) b.velocity /= G;
    if(radius_div_g) b.radius /= G;

    res.push_back(b);
  }

  return {res, set};
}
