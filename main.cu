#include <iostream>

#include "rendering/shader.cuh"
#include "rendering/camera.cuh"
#include "rendering/object.cuh"
#include "rendering/line.cuh"
#include "rendering/gl_wrapper.cuh"
#include "rendering/fps_counter.cuh"
#include "cuda_gl_bridge.cuh"
#include "simulation/stepper.cuh"
#include "loader.cuh"

template <bool history, bool moving_cam>
void render_loop(
  cu_sim::gl_wrapper &render, const cu_sim::settings &settings, cu_sim::stepper &simulator, const cu_sim::shader &shader,
  cu_sim::moving_cam &camera, const cu_sim::object &object, const cu_sim::shader &line_shader, const cu_sim::hist_line &line
) {
  static size_t frame = 0;
  static bool step = true;

  render.clear();
  if(render.toggle_progress()) step = !step;

  if(step) {
    if constexpr(history)
      simulator.step(render.delta_time() * settings.time_scale, frame);
    else
      simulator.step_no_history(render.delta_time() * settings.time_scale, frame);
  }

  if(render.is_zoom_in()) {
    camera.camera.fov_y *= 0.9f;
  }
  if(render.is_zoom_out()) {
    camera.camera.fov_y *= 1.11f;
  }

  if constexpr(moving_cam) camera.interpolate(render.delta_time());

  shader.enable();
  shader.set_m4(1, camera.camera.view_matrix()); // view
  shader.set_m4(2, camera.camera.projection_matrix()); // projection
  shader.set_vec3(4, camera.camera.eye); // light pos
  shader.set_vec3(5, glm::vec3{ 0.7f, 0.65f, 0.6f }); // light color
  shader.set_float(6, 1.0f); // ambient factor
  shader.set_vec3(7, camera.camera.eye); // view pos
  shader.set_float(8, 16.0f); // phong exponent
  object.draw();

  if constexpr (history) {
    line_shader.enable();
    line_shader.set_m4(1, camera.camera.view_matrix());
    line_shader.set_m4(2, camera.camera.projection_matrix());
    line.draw(line_shader);
  }

  cu_sim::fps_counter::get().draw();

  render.frame();
  ++frame;
}

int main(const int argc, const char **argv) {
  if(argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <path to INI>\n"
              << "\n"
              << " -> INI spec:\n"
              << "    -> 1 section ([object name]) per body in the simulation\n"
              << "    -> Each section must have the following keys:\n"
              << "       -> position: vec3 (three floating-point literals, separated by spaces)\n"
              << "       -> velocity: vec3\n"
              << "       -> mass: float\n"
              << "    -> Optional keys per section:\n"
              << "       -> radius: float (default: 0.1); specifies the rendering radius\n"
              << "       -> color: vec3 (default: 0.22 0.22 0.22); specifies the color of the body in the rendering\n"
              << "       -> mass_div_g: bool (default: false); specifies whether or not to divide the mass by G\n"
              << "       -> pos_div_g: bool (default: false); specifies whether or not to divide the position by G\n"
              << "       -> vel_div_g: bool (default: false); specifies whether or not to divide the velocity by G\n"
              << "       -> radius_div_g: bool (default: false); specifies whether or not to divide the radius by G\n"
              << "    -> The G value used is " << cu_sim::G << "\n"
              << "    -> All other keys are ignored.\n"
              << "    -> Optional settings section [SETTINGS]:\n"
              << "       -> time_scale: float (default 0.001)\n"
              << "       -> history: int (default 100); the amount of previous steps for the trailing line\n"
              << "       -> history_skip: int (default 100); the amount of steps before history is updated\n"
              << "       -> history_render: bool (default true); whether or not to render history lines\n"
              << "       -> eye: vec3 (default 0 0 7); the location of the camera\n"
              << "       -> focus: vec3 (default 0 0 0); the location the camera looks at\n"
              << "       -> moving_cam: bool (default false); toggles a randomly moving camera, always focused on focus\n"
              << "       -> cam_time_scale: float (default 0.1); speed of the (randomly moving) camera\n"
    ;
    return -1;
  }

  try {
    const auto [bodies, settings] = cu_sim::load_bodies(argv[1]);
    const cu_sim::shader shader("shaders/vertex.glsl", "shaders/fragment.glsl");
    const cu_sim::shader line_shader("shaders/vertex_line.glsl", "shaders/fragment_line.glsl");
    const cu_sim::object object("assets/body.obj", bodies.size());
    const cu_sim::hist_line line(settings.history_size, bodies.size());
    cu_sim::camera cam(settings.eye, settings.focus, {0, 1, 0});
    cam.fov_y = 90.0f;

    cu_sim::moving_cam camera(cam, settings.cam_time_scale);

    cu_sim::stepper simulator(object.cuda_buffers(), line.cuda_handles(), bodies, settings.history_size, settings.history_skip);

    auto &render = cu_sim::gl_wrapper::get();
    render.clear();
    render.frame();

#define RENDER_LOOP(hist, cam) do { \
    render_loop<hist, cam>(render, settings, simulator, shader, camera, object, line_shader, line); \
  } while(!render.should_close())

    if(settings.enable_history) {
      if(settings.moving_cam) RENDER_LOOP(true, true);
      else RENDER_LOOP(true, false);
    }
    else {
      if(settings.moving_cam) RENDER_LOOP(false, true);
      else RENDER_LOOP(false, false);
    }

#undef RENDER_LOOP
  }
  catch(const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
