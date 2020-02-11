#version 450
layout (push_constant) uniform PushConsts {
  float time;
} push;

layout(location = 0) in vec3 frag_color;
layout(location = 0) out vec4 target0;

void main() {
    target0 = vec4(frag_color, 1.0) * vec4(push.time, push.time, push.time, 1.0);
}
