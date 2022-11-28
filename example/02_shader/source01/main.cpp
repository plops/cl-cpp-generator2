// try to learn how to write shaders based on
// https://www.youtube.com/watch?v=pmS-F6RJhAk
float Hash21(vec2 p) {
  p = fract(((p) * (vec2((123.222222340), (132.3230)))));
  (p) += (dot(p, ((p) + ((223.120)))));
  return fract(((p.x) * (p.y)));
}
float Length(vec2 p, float k) {
  // minkowsky distance
  ;
  p = abs(p);
  return pow(((pow(p.x, k)) + (pow(p.y, k))), (((1.0)) / (k)));
}
vec4 Truchet(vec2 p, vec3 col, float curve, float thickness, float pattern) {
  vec2 id = floor(p);
  float n = Hash21(id);
  vec3 col2 = vec3((0.));
  float depth2 = (0.);
  float d = (0.);
  p = ((fract(p)) - ((0.50)));
  if ((n) < ((0.50))) {
    p.xy = vec2(p.x, -p.y);
  }
  {
    // circle around top-right
    ;
    vec2 cp = ((p) + ((-0.50)));
    float a = atan(cp.y, cp.x);
    float depth = (((0.50)) + ((((0.50)) * (cos((((2.0)) * (a)))))));
    float contour = smoothstep(
        (1.00e-2), (-1.00e-2),
        ((abs(((Length(((p) + ((-0.50))), curve)) - ((0.50))))) - (thickness)));
    (depth2) += (((depth) * (contour)));
    (col2) += (((mix((0.20), (1.0), depth)) * (col) * (contour)));
    col2 *=
        ((((1.0)) +
          ((((0.30)) * (pattern) *
            (sin((((((1.0)) * ((30.)) * (a))) +
                  ((((1.00e+2)) *
                    (((abs(((Length(((p) + ((-0.50))), curve)) - ((0.50))))) -
                      (thickness))))) +
                  ((((-5.0)) * (iTime))))))))));
  }
  {
    // circle around btm-left
    ;
    vec2 cp = ((p) + ((0.50)));
    float a = atan(cp.y, cp.x);
    float depth = (((0.50)) + ((((0.50)) * (cos((((2.0)) * (a)))))));
    float contour = smoothstep(
        (1.00e-2), (-1.00e-2),
        ((abs(((Length(((p) + ((0.50))), curve)) - ((0.50))))) - (thickness)));
    (depth2) += (((depth) * (contour)));
    (col2) += (((mix((0.20), (1.0), depth)) * (col) * (contour)));
    col2 *=
        ((((1.0)) +
          ((((0.30)) * (pattern) *
            (sin((((((-1.0)) * ((30.)) * (a))) +
                  ((((1.00e+2)) *
                    (((abs(((Length(((p) + ((0.50))), curve)) - ((0.50))))) -
                      (thickness))))) +
                  ((((-5.0)) * (iTime))))))))));
  }
  if ((1) == (1)) {
    // DEBUG: visualize edge of tile
    if (((((0.490)) < (p.x)) || ((p.x) < ((-0.490))) || (((0.490)) < (p.y)) ||
         ((p.y) < ((-0.490))))) {
      return vec4((1.0), (1.0), (1.0), (1.0));
    }
  }
  return vec4(col2, depth2);
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  vec2 uv =
      ((((fragCoord) - ((((0.50)) * (iResolution.xy))))) / (iResolution.y));
  vec3 col = vec3((0.));
  float cd = length(uv);
  float w = mix((0.10), (1.00e-2), smoothstep((0.), (0.50), cd));
  uv *= ((3.0));
  vec4 t1 = Truchet(uv, vec3((1.0), (0.), (0.)), (2.0), w, (1.0));
  vec4 t2 =
      Truchet(((uv) + ((0.50))), vec3((0.), (1.0), (0.)), (1.0), (0.10), (0.));
  if ((t2.a) < (t1.a)) {
    (col) += (t1.rgb);
  }
  if ((t1.a) < (t2.a)) {
    (col) += (t2.rgb);
  }
  fragColor = vec4(col, (1.0));
}