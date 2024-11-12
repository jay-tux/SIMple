#version 460 core
layout (location = 0) in vec2 pos; // = tex coords

layout (location = 1) uniform vec2 position;
layout (location = 2) uniform float aspect;
layout (location = 3) uniform float scale;
layout (location = 4) uniform int num;
layout (location = 5) uniform sampler2D tex;
layout (location = 6) uniform vec4 rgba;

out vec2 tex_out;

void main() {
    vec4 base = vec4(position + vec2(1.0f, aspect) * scale * pos, 0.0, 1.0);
    gl_Position = base;
    vec2 glyph_zero = vec2((num % 10) / 10.0f, 0.0f);
    tex_out = glyph_zero + pos * vec2(0.1f, 1.0f);
    tex_out.y = -tex_out.y;
}
