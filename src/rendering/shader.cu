//
// Created by jay on 9/30/24.
//

#include <sstream>
#include <fstream>
#include <filesystem>
#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>

#include "shader.cuh"
#include "path.cuh"
#include "gl_wrapper.cuh"

using namespace cu_sim;

shader::shader(const std::string &vertex_path, const std::string &fragment_path) {
  gl_wrapper::force_initialized();

  auto from_file = [](const std::string &path, const std::string &type) {
    const auto pth = binary_path() / path;
    std::ifstream input(pth);
    if(!input.is_open()) {
      throw std::runtime_error("Failed to open shader file " + pth.string());
    }
    std::stringstream source_strm;
    source_strm << input.rdbuf();
    input.close();
    const auto source = source_strm.str();

    auto id = glCreateShader(type == "vertex" ? GL_VERTEX_SHADER : GL_FRAGMENT_SHADER);
    const char *src = source.c_str();
    glShaderSource(id, 1, &src, nullptr);
    glCompileShader(id);
    int success;
    glGetShaderiv(id, GL_COMPILE_STATUS, &success);
    if(!success) {
      glGetShaderiv(id, GL_INFO_LOG_LENGTH, &success);
      auto *info_log = new char[success];
      glGetShaderInfoLog(id, success, nullptr, info_log);
      glDeleteShader(id);
      throw std::runtime_error("Failed to compile " + type + " shader " + pth.string() + ": " + info_log);
    }

    return id;
  };

  vertex_shader = from_file(vertex_path, "vertex");
  fragment_shader = from_file(fragment_path, "fragment");

  shader_id = glCreateProgram();
  glAttachShader(shader_id, vertex_shader);
  glAttachShader(shader_id, fragment_shader);
  glLinkProgram(shader_id);
  int success;
  glGetProgramiv(shader_id, GL_LINK_STATUS, &success);
  if(!success) {
    glGetProgramiv(shader_id, GL_INFO_LOG_LENGTH, &success);
    std::string info_log(success, '\0');
    glGetProgramInfoLog(shader_id, success, nullptr, info_log.data());
    glDeleteProgram(shader_id);
    throw std::runtime_error("Failed to link shader program: " + info_log);
  }
}

void shader::enable() const {
  glUseProgram(shader_id);
}

void shader::set_m4(const int name, const glm::mat4 &value) const {
  enable();
  glUniformMatrix4fv(name, 1, GL_FALSE, glm::value_ptr(value));
}

void shader::set_float(const int name, const float value) const {
  enable();
  glUniform1f(name, value);
}

void shader::set_int(const int name, const int value) const {
  enable();
  glUniform1i(name, value);
}

void shader::set_vec2(const int name, const glm::vec2 &value) const {
  enable();
  glUniform2fv(name, 1, glm::value_ptr(value));
}

void shader::set_vec3(const int name, const glm::vec3 &value) const {
  enable();
  glUniform3fv(name, 1, glm::value_ptr(value));
}

void shader::set_vec4(const int name, const glm::vec4 &value) const {
  enable();
  glUniform4fv(name, 1, glm::value_ptr(value));
}

shader::~shader() {
  glDeleteShader(vertex_shader);
  glDeleteShader(fragment_shader);
  glDeleteProgram(shader_id);
}