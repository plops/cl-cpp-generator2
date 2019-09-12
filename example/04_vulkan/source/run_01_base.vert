#version 450

layout(location = 0) out vec3 fragColor;
vec2 positions[] = vec2[](vec2((0.0e+0), (-5.e-1)), vec2((5.e-1), (5.e-1)),
                          vec2((-5.e-1), (5.e-1)));
vec3 colors[] = vec3[](vec3(1, (0.0e+0), (0.0e+0)), vec3((0.0e+0), 1, (0.0e+0)),
                       vec3((0.0e+0), (0.0e+0), 1));
void main() {
  gl_Position = vec4(positions[gl_VertexIndex], (0.0e+0), 1);
  fragColor = colors[gl_VertexIndex];
}
// vertex shader end ;