#version 450

vec2 positions[3] = (vec2)((vec2((0.0e+0), (-5.e-1)), vec2((5.e-1), (5.e-1)),
                            vec2((-5.e-1), (5.e-1))));
void main() { gl_Position = vec4(positions[gl_VertexIndex], (0.0e+0), 1); };