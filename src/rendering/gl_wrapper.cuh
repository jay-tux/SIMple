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

class fps_fsm {
public:
  fps_fsm &operator<<(const float time) {
    for(int i = 0; i < 3; i++) {
      times[i] = times[i + 1];
    }
    times[3] = time;
    return *this;
  }

  constexpr float delta() const { return times[1] - times[0]; }
  constexpr float fps() const { return 3.0f / (times[3] - times[0]); }
  constexpr float time() const { return times[1]; }
private:
  float times[4] { 0, 0 };
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
  constexpr float time() const { return fps_.time(); }
  constexpr float delta_time() const { return fps_.delta(); }
  constexpr float fps() const { return fps_.fps(); }
  bool should_close() const;
  constexpr bool toggle_progress() const {
    return play_pause.state_release();
  }
  constexpr bool is_zoom_in() const { return zoom_in.state_release(); }
  constexpr bool is_zoom_out() const { return zoom_out.state_release(); }

  ~gl_wrapper();
private:
  gl_wrapper();

  GLFWwindow *win;
  key_fsm play_pause;
  key_fsm exit;
  key_fsm zoom_in;
  key_fsm zoom_out;
  fps_fsm fps_;
};
}

#endif //GL_WAPPER_CUH
