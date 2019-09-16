#version 450

layout(location = 0) out vec4 target0;

void main() {
    target0 = vec4(1.0, 0.0, 0.0, 1.0); // texture(sampler2D(u_texture, u_sampler), v_uv) * color_dat.color;
}
