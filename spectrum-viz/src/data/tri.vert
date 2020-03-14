#version 450
layout (push_constant) uniform PushConsts {
  mat4 view_proj;
} push;

layout(location = 0) in vec3 vert_pos;
layout(location = 1) in vec3 vert_col;
layout(location = 2) in vec2 vert_uv;
layout(location = 3) in vec4 model_col1;
layout(location = 4) in vec4 model_col2;
layout(location = 5) in vec4 model_col3;
layout(location = 6) in vec4 model_col4;

layout(location = 0) out vec3 frag_col;
layout(location = 1) out vec2 frag_uv;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    mat4 model = mat4(
        model_col1,
        model_col2,
        model_col3,
        model_col4);
    gl_Position = push.view_proj * model * vec4(vert_pos, 1.0);
    frag_col = vert_col;
    frag_uv = vert_uv;
}
