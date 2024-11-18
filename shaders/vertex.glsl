#version 460 core

layout(location = 0) in vec3 pos; // x y z
layout(location = 1) in vec2 uv; // u v
layout(location = 2) in vec3 normal; // nx ny nz
layout(location = 3) in vec3 offset;
layout(location = 4) in float radius;
layout(location = 5) in vec3 in_color;
layout(location = 6) in vec3 spec_color; // r g b

layout(location = 1) uniform mat4 view;
layout(location = 2) uniform mat4 projection;
layout(location = 4) uniform vec3 light_pos;
layout(location = 5) uniform vec3 light_color;
layout(location = 6) uniform float ambient;
layout(location = 7) uniform vec3 view_pos;
layout(location = 8) uniform float phong;

out vec3 color;
out vec3 f_spec;
out vec3 f_normal;
out vec3 f_pos;

void main() {
    vec4 model = vec4(radius * pos, 1.0) + vec4(offset, 0.0f);
    gl_Position = projection * view * model;
    color = in_color;
    f_normal = normal;
    f_pos = gl_Position.xyz;
    f_spec = spec_color;
}
