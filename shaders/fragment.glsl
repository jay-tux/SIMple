#version 460 core

layout(location = 4) uniform vec3 light_pos;
layout(location = 5) uniform vec3 light_color;
layout(location = 6) uniform float ambient;
layout(location = 7) uniform vec3 view_pos;
layout(location = 8) uniform float phong;

in vec3 color;
in vec3 f_spec;
in vec3 f_normal;
in vec3 f_pos;
out vec4 frag_color;

void main() {
    vec3 n_normal = normalize(f_normal);
    vec3 ambient_col = ambient * light_color * color;

    vec3 light_dir = normalize(light_pos - f_pos);

    float diffuse = max(dot(n_normal, light_dir), 0.0f);
    vec3 diff_col = diffuse * light_color * color;

    vec3 refl = normalize(reflect(-light_dir, f_normal));
    float specular = pow(max(dot(normalize(view_pos - f_pos), refl), 0.0f), phong);
    vec3 spec_col = specular * light_color * f_spec;

    frag_color = vec4(ambient_col + diff_col + spec_col, 1.0);
}
