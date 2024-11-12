//
// Created by jay on 9/30/24.
//

#include <vector>
#include <glad/glad.h>

#include "line.cuh"
#include "gl_wrapper.cuh"

using namespace cu_sim;

hist_line::hist_line(const size_t history_size, const size_t instances) : history_size{history_size}, instance_count{instances} {
  gl_wrapper::force_initialized();

  std::vector<float> history_vertices;
  history_vertices.reserve(history_size);
  for(size_t i = 0; i < history_size; i++) {
    history_vertices.push_back(i);
  }

  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, history_size * sizeof(float), history_vertices.data(), GL_DYNAMIC_DRAW);
  glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(0);

  glGenBuffers(1, &cuda.color);
  glBindBuffer(GL_ARRAY_BUFFER, cuda.color);
  glBufferData(GL_ARRAY_BUFFER, 3 * instances * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(1);

  glBindVertexArray(0);

  glGenBuffers(1, &cuda.history);
  glBindBuffer(GL_ARRAY_BUFFER, cuda.history);
  glBufferData(GL_ARRAY_BUFFER, 3 * instances * history_size * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

  glGenTextures(1, &tbo);
}

void hist_line::draw(const shader &s) const {
  glBindVertexArray(vao);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_BUFFER, tbo);
  glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, cuda.history);
  s.set_int(3, 1);
  s.set_int(4, history_size);

  glDrawArraysInstanced(GL_LINE_STRIP, 0, history_size, instance_count);
}

hist_line::~hist_line() {
  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(1, &vbo);
  glDeleteBuffers(1, &cuda.history);
  glDeleteBuffers(1, &cuda.color);
  glDeleteBuffers(1, &tbo);
}