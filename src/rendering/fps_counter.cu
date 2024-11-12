//
// Created by jay on 11/12/24.
//

#include <glad/glad.h>
#include <stb_image.h>
#include <stdexcept>

#include "fps_counter.cuh"
#include "gl_wrapper.cuh"

using namespace cu_sim;

namespace {
constexpr float fps_vbo[8] = {
  // x     y
  0.0f, 0.0f,
  1.0f, 0.0f,
  1.0f, 1.0f,
  0.0f, 1.0f,
};

constexpr int fps_idx[6] = {
  0, 1, 2,
  0, 2, 3
};
}

fps_counter::fps_counter() : s("shaders/fps_vert.glsl", "shaders/fps_frag.glsl") {
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(float), fps_vbo, GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
  glGenBuffers(1, &ebo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6 * sizeof(int), fps_idx, GL_STATIC_DRAW);
  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  int w; int h;
  unsigned char *img = stbi_load("shaders/digits.png", &w, &h, nullptr, 4);
  if(img == nullptr) {
    throw std::runtime_error("Failed to load font texture0");
  }

  glGenTextures(1, &tex_id);
  glBindTexture(GL_TEXTURE_2D, tex_id);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
  glGenerateMipmap(GL_TEXTURE_2D);

  stbi_image_free(img);
}

fps_counter &fps_counter::get() {
  static fps_counter counter;
  return counter;
}

#include <iostream>

void fps_counter::draw() const {
  glBindVertexArray(vao);
  s.enable();
  const auto fps = std::max(std::min(static_cast<int>(gl_wrapper::get().fps()), 99), 0);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, tex_id);
  const int fps_parts[2] { fps / 10, fps % 10 };
  s.set_float(2, aspect);
  s.set_float(3, 0.02f);
  s.set_int(5, 0);
  s.set_vec4(6, {0.5, 0.5, 1.0, 1.0});

  s.set_vec2(1, {-0.98, 0.85});
  s.set_int(4, fps_parts[0]);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

  s.set_vec2(1, {-0.965, 0.85});
  s.set_int(4, fps_parts[1]);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

  glBindVertexArray(0);

  std::cout << "FPS: " << fps << "\n";
}

fps_counter::~fps_counter() {
  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(1, &vbo);
  glDeleteBuffers(1, &ebo);
  glDeleteTextures(1, &tex_id);
}
