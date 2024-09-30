//
// Created by jay on 9/30/24.
//

#include <stdexcept>
#include <glad/glad.h>

#include "gl_wrapper.cuh"

using namespace cu_sim;

gl_wrapper::gl_wrapper() {
  if(!glfwInit()) {
    throw std::runtime_error("Failed to initialize GLFW");
  }

  win = glfwCreateWindow(1920, 1080, "cuSIM", nullptr, nullptr);
  if(win == nullptr) {
    glfwTerminate();
    throw std::runtime_error("Failed to create window");
  }
  glfwMakeContextCurrent(win);

  if(!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
    glfwTerminate();
    throw std::runtime_error("Failed to initialize GLAD");
  }

  glViewport(0, 0, 1920, 1080);
  glEnable(GL_DEPTH_TEST);
  glClearColor(0, 0, 0, 0);

  last_time = glfwGetTime();
}

gl_wrapper &gl_wrapper::get() {
  static gl_wrapper instance;
  return instance;
}

void gl_wrapper::force_initialized() {
  get();
}

// ReSharper disable once CppMemberFunctionMayBeStatic
void gl_wrapper::clear() const {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

float gl_wrapper::aspect() const {
  int w, h;
  glfwGetWindowSize(win, &w, &h);
  return static_cast<float>(w) / h;
}

void gl_wrapper::frame() {
  glfwSwapBuffers(win);
  glfwPollEvents();
  last_time = glfwGetTime();
  exit.shift_in(glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS);
  play_pause.shift_in(glfwGetKey(win, GLFW_KEY_SPACE) == GLFW_PRESS);
}

float gl_wrapper::time() const {
  return last_time;
}

float gl_wrapper::delta_time() const {
  return glfwGetTime() - last_time;
}

bool gl_wrapper::should_close() const {
  return glfwWindowShouldClose(win) || exit.state_release();
}

gl_wrapper::~gl_wrapper() {
  glfwDestroyWindow(win);
  glfwTerminate();
}

void key_fsm::shift_in(const bool is_pressed) {
  // shift old state away, insert new state at end
  state = (state << 1) | (is_pressed ? 1 : 0);
}
