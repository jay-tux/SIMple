#include <iostream>

#include "rendering/shader.cuh"
#include "rendering/camera.cuh"
#include "rendering/object.cuh"
#include "rendering/gl_wrapper.cuh"
#include "cuda_gl_bridge.cuh"
#include "simulation/stepper.cuh"
#include "loader.cuh"

__global__ void micro_kernel(float3 *pos, float *radius, float3 *color, const size_t count, const float time) {
  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx >= count) return;

  if(idx % 2 == 0) {
    pos[idx].x = sinf(time);
    pos[idx].y = 0;
    pos[idx].z = 0;

    radius[idx] = abs(cosf(time));

    color[idx].x = 1.0f - abs(tanhf(sinf(time)));
    color[idx].y = abs(tanhf(cosf(time)));
    color[idx].z = 0.0f;
  }
  else {
    pos[idx].x = 0;
    pos[idx].y = cosf(time);
    pos[idx].z = 0;

    radius[idx] = abs(sinf(time));

    color[idx].x = 0.0f;
    color[idx].y = abs(tanhf(sinf(time)));
    color[idx].z = 1.0f - abs(tanhf(cosf(time)));
  }
}

constexpr static float time_scale = 2e2f;

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
              << "    -> All other keys are ignored.\n";
    return -1;
  }

  try {
    const auto bodies = cu_sim::load_bodies(argv[1]);
    const cu_sim::shader shader("shaders/vertex.glsl", "shaders/fragment.glsl");
    const cu_sim::object object("assets/body.obj", bodies.size());
    cu_sim::camera camera({0, 0, 7}, {0, 0, 0}, {0, 1, 0});
    camera.fov_y = 90.0f;

    cu_sim::stepper simulator(object.cuda_buffers(), bodies);

    auto &render = cu_sim::gl_wrapper::get();
    render.clear();
    render.frame();
    bool step = true;
    while(!render.should_close()) {
      render.clear();
      if(render.toggle_progress()) step = !step;

      if(step)
        simulator.step(render.delta_time() * time_scale);

      shader.enable();
      shader.set_m4(1, camera.view_matrix());
      shader.set_m4(2, camera.projection_matrix());
      object.draw();

      render.frame();
    }
  }
  catch(const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
