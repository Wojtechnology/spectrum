#version 450
layout (push_constant) uniform PushConsts {
  mat4 mvp;
} push;

layout(location = 0) in vec3 vert_pos;
layout(location = 1) in vec3 vert_col;
layout(location = 2) in vec2 vert_uv;
layout(location = 0) out vec3 frag_col;
layout(location = 1) out vec2 frag_uv;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    gl_Position = push.mvp * vec4(vert_pos, 1.0);
    frag_col = vert_col;
    frag_uv = vert_uv;
}
