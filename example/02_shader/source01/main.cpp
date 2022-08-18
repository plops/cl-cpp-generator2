// try to learn how to write shaders based on
// https://www.youtube.com/watch?v=pmS-F6RJhAk
float Hash21(vec2 p) {
  p = fract(((p) * (vec2((123.222222340), (132.3230)))));
  (p) += (dot(p, ((p) + ((223.120)))));
  return fract(((p.x) * (p.y)));
}
vec4 Truchet(vec2 p, vec3 col) {
  vec2 id = floor(p);
  float n = Hash21(id);
  vec3 col2 = vec3((0.));
  float d = (0.);
  p = ((fract(p)) - ((0.50)));
  if ((n) < ((0.50))) {
    p.xy = vec2(p.x, -p.y);
  }
  // circle around top-right
  ;
  (col2) +=
      (((col) * (smoothstep((1.00e-2), (-1.00e-2),
                            ((abs(((length(((p) + ((-0.50))))) - ((0.50))))) -
                             ((5.00e-2)))))));
  // circle around btm-left
  ;
  (col2) +=
      (((col) * (smoothstep((1.00e-2), (-1.00e-2),
                            ((abs(((length(((p) + ((0.50))))) - ((0.50))))) -
                             ((5.00e-2)))))));
  if (((((0.490)) < (p.x)) || ((p.x) < ((-0.490))) || (((0.490)) < (p.y)) ||
       ((p.y) < ((-0.490))))) {
    (col2) += ((1.0));
  }
  return vec4(col2, d);
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  vec2 uv =
      ((((fragCoord) - ((((0.50)) * (iResolution.xy))))) / (iResolution.y));
  vec3 col = vec3((0.));
  uv *= ((3.0));
  vec4 t1 = Truchet(uv, vec3((1.0), (0.), (0.)));
  col = t1.rgb;
  fragColor = vec4(col, (1.0));
}