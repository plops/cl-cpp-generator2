#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) out vec4 outColor;
layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
void main() { outColor = vec4(fragTexCoord, (0.0e+0f), (1.e+0f)); }