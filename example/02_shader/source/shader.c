float map(in vec3 pos) {
  auto d = ((length(pos)) - ((2.5e-1)));
  ;
  return d;
  ;
};
vec3 calcNormal(in vec3 pos) {
  return normalize(vec3(((map(((pos) + (e.xyy)))) - (map(((pos) - (e.xyy))))),
                        ((map(((pos) + (e.xyy)))) - (map(((pos) - (e.xyy))))),
                        ((map(((pos) + (e.xyy)))) - (map(((pos) - (e.xyy)))))));
};
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  vec2 p =
      (((((2.e+0)) * (((fragCoord) - (iResolution.xy))))) / (iResolution.y));
  ;
  vec3 ro = vec3((0.0e+0), (0.0e+0), (2.e+0));
  ;
  vec3 rd = normalize(vec3(p, (-1.4999999999999997e+0)));
  ;
  vec3 col = vec3((0.0e+0));
  ;
  float tt = (0.0e+0);
  ;
  for (i = 0;; i < 100; (i)++) {
    vec3 pos = ((ro) + (((tt) * (rd))));
    ;
    float h = map(pos);
    ;
    if (h < (1.0000000474974513e-3)) {
      break;
      ;
    };
    (tt) += (h);
    if ((2.e+1) < t) {
      break;
      ;
    };
    ;
  };
  if (t < (2.e+1)) {
    vec3 pos = ((ro) + (((t) * (rd))));
    ;
    vec3 nor = calcNormal(pos);
    ;
    vec3 sun_dir = normalize(vec3((8.00000011920929e-1), (4.000000059604645e-1),
                                  (-2.0000000298023224e-1)));
    ;
    float dif = clamp(nor.sim_dir, (0.0e+0), (1.e+0));
    ;
    col = ((vec3((1.e+0), (8.e+0), (5.e-1))) * (dif));
    ;
    ;
    ;
  };
  fragColor = vec4(col, (1.e+0));
  ;
  ;
};