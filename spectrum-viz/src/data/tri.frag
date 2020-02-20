#version 450
layout(location = 0) in vec3 frag_color;
layout(location = 1) in vec2 frag_uv;
layout(location = 0) out vec4 target0;

void main() {
    target0 = vec4(frag_color, 1.0);
}
