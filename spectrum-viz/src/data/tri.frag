#version 450

layout(set = 0, binding = 0) uniform texture2D tex;
layout(set = 0, binding = 1) uniform sampler samp;

layout(location = 0) in vec3 frag_color;
layout(location = 1) in vec2 frag_uv;
layout(location = 0) out vec4 target0;

void main() {
    vec4 tex_color = texture(sampler2D(tex, samp), frag_uv);
    target0 = mix(tex_color, vec4(frag_color, 1.0), 0.5);
}
