// try to learn how to write shaders based on
// https://www.youtube.com/watch?v=pmS-F6RJhAk
vec4 Truchet(vec2 p) {
  p = ((fract(p)) - ((0.50)));
  vec3 col = vec3((0.));
  float d = (0.);
  {
    float cd = length(((p) + ((-0.50))));
    // cd .. distance to top-right
    ;
    (col) +=
        (smoothstep((0.10), (-0.10), ((abs(((cd) - ((0.50))))) - ((5.00e-2)))));
  }
  {
    float cd = length(((p) + ((0.50))));
    // cd .. distance to btm-left
    ;
    (col) +=
        (smoothstep((0.10), (-0.10), ((abs(((cd) - ((0.50))))) - ((5.00e-2)))));
  }
  if (((((0.490)) < (p.x)) || ((p.x) < ((-0.490))) || (((0.490)) < (p.y)) ||
       ((p.y) < ((-0.490))))) {
    (col) += ((1.0));
  }
  return vec4(col, d);
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  vec2 uv =
      ((((fragCoord) - ((((0.50)) * (iResolution.xy))))) / (iResolution.y));
  vec3 col = vec3((0.));
  uv *= ((3.0));
  vec4 t1 = Truchet(uv);
  col = t1.rgb;
  fragColor = vec4(col, (1.0));
}