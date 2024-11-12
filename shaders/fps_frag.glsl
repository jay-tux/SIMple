#version 460

in vec2 tex_out;

layout (location = 5) uniform sampler2D tex;
layout (location = 6) uniform vec4 rgba;

out vec4 res;

void main() {
    res = texture(tex, tex_out) * rgba;
    if(res.a < 0.5) discard;
}