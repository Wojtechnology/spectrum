#version 450
layout (push_constant) uniform PushConsts {
  float time;
} push;

layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec3 a_color;
layout(location = 0) out vec3 frag_color;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    gl_Position = vec4(a_pos.x * push.time, a_pos.y * push.time, 0.0, 1.0);
    frag_color = a_color;
}
