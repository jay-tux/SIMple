#version 460 core

layout (location = 0) in float _unused_;
layout (location = 1) in vec3 in_color;

layout (location = 1) uniform mat4 view;
layout (location = 2) uniform mat4 projection;
layout (location = 3) uniform samplerBuffer tbo;
layout (location = 4) uniform int history_size;

out vec3 color;

void main() {
    int idx = history_size * gl_InstanceID + gl_VertexID;
    vec3 actual = texelFetch(tbo, idx).xyz;
    gl_Position = projection * view * vec4(actual, 1.0);
    color = in_color;
}
