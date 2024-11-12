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

  cuda.history.resize(history_size);
  glGenBuffers(history_size, cuda.history.data());
  for(size_t i = 0; i < history_size; i++) {
    glBindBuffer(GL_ARRAY_BUFFER, cuda.history[i]);
    glBufferData(GL_ARRAY_BUFFER, 3 * instances * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(i + 1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glVertexAttribDivisor(i + 1, 1);
    glEnableVertexAttribArray(i + 1);
  }

  glGenBuffers(1, &cuda.color);
  glBindBuffer(GL_ARRAY_BUFFER, cuda.color);
  glBufferData(GL_ARRAY_BUFFER, 3 * instances * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(history_size + 1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(history_size + 1);
}

void hist_line::draw() const {
  glBindVertexArray(vao);
  glDrawArraysInstanced(GL_LINE_STRIP, 0, history_size, instance_count);
}

hist_line::~hist_line() {
  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(1, &vbo);
  glDeleteBuffers(cuda.history.size(), cuda.history.data());
  glDeleteBuffers(1, &cuda.color);
}