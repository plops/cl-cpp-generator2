// Inigo Quilez: LIVE Shader Deconstruction :: happy jumping
// https://www.youtube.com/watch?v=Cfe5UQ-1L9Q ;
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  vec2 p =
      (((((((2.e+0)) * (fragCoord))) - (iResolution.xy))) / (iResolution.y));
  vec3 col = vec3((0.0e+0));
  float f = ((length(p)) * ((5.e-1)));
  col = vec3(f, f, f);
  fragColor = vec4(col, (1.e+0));
}