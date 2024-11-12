#version 460 core

layout (location = 0) in float _unused_;
layout (location = 1) in vec3 history0;
layout (location = 2) in vec3 history1;
layout (location = 3) in vec3 history2;
layout (location = 4) in vec3 history3;
layout (location = 5) in vec3 history4;
layout (location = 6) in vec3 history5;
layout (location = 7) in vec3 history6;
layout (location = 8) in vec3 history7;
layout (location = 9) in vec3 history8;
layout (location = 10) in vec3 history9;
layout (location = 11) in vec3 history10;
layout (location = 12) in vec3 history11;
layout (location = 13) in vec3 history12;
layout (location = 14) in vec3 history13;
layout (location = 15) in vec3 in_color;

layout (location = 1) uniform mat4 view;
layout (location = 2) uniform mat4 projection;

out vec3 color;

void main() {
    vec3 hist[14] = {
        history0, history1, history2, history3, history4, history5, history6, history7, history8, history9,
        history10, history11, history12, history13
    };
    int idx = clamp(gl_VertexID, 0, 9);
    vec3 actual = hist[idx].xyz;
    gl_Position = projection * view * vec4(actual, 1.0);
    color = in_color;
}
