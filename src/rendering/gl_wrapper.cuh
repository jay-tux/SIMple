//
// Created by jay on 9/30/24.
//

#ifndef GL_WAPPER_CUH
#define GL_WAPPER_CUH

#include <GLFW/glfw3.h>

namespace cu_sim {
class key_fsm {
public:
  void shift_in(bool is_pressed);
  constexpr bool state_off() const {
    return useful_bits() == 0b00;
  }
  constexpr bool state_press() const {
    return useful_bits() == 0b01;
  }
  constexpr bool state_hold() const {
    return useful_bits() == 0b11;
  }
  constexpr bool state_release() const {
    return useful_bits() == 0b10;
  }

private:
  constexpr uint8_t useful_bits() const {
    return state & 0b11;
  }

  uint8_t state = 0b00;
};

class gl_wrapper {
public:
  gl_wrapper(const gl_wrapper &) = delete;
  gl_wrapper(gl_wrapper &&) = delete;
  gl_wrapper &operator=(const gl_wrapper &) = delete;
  gl_wrapper &operator=(gl_wrapper &&) = delete;

  static gl_wrapper &get();
  static void force_initialized();

  void clear() const;
  void frame();
  float aspect() const;
  float time() const;
  float delta_time() const;
  inline float fps() const { return 1.0f / delta_time(); }
  bool should_close() const;
  constexpr bool toggle_progress() const {
    return play_pause.state_release();
  }

  ~gl_wrapper();
private:
  gl_wrapper();

  GLFWwindow *win;
  float last_time;
  key_fsm play_pause;
  key_fsm exit;
};
}

#endif //GL_WAPPER_CUH
