#version 460 core

layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 offset;
layout(location = 4) in float radius;
layout(location = 5) in vec3 in_color;

layout(location = 1) uniform mat4 view;
layout(location = 2) uniform mat4 projection;

out vec3 color;

void main() {
    vec4 model = vec4(radius * pos, 1.0) + vec4(offset, 0.0f);
    gl_Position = projection * view * model;
    color = in_color;
}
