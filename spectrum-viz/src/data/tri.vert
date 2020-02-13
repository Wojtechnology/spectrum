#version 450
layout (push_constant) uniform PushConsts {
  float time;
} push;

layout(location = 0) in vec3 vert_pos;
layout(location = 1) in vec3 vert_col;
layout(location = 2) in vec2 vert_uv;
layout(location = 0) out vec3 frag_color;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    gl_Position = vec4(vert_pos.x * push.time, vert_pos.y * push.time, vert_pos.z, 1.0);
    frag_color = vert_col;
}
