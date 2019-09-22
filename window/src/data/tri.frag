#version 450
layout (push_constant) uniform PushConsts {
  float time;
} push;

layout(location = 0) in vec3 frag_color;
layout(location = 0) out vec4 target0;

void main() {
    float time01 = -0.9 * abs(sin(push.time * 0.9)) + 0.9;
    target0 = vec4(frag_color, 1.0) * vec4(1.0, 1.0, time01, 1.0);
}