float map(in vec3 pos) {
  float d = ((length(pos)) - ((2.5e-1)));
  ;
  float d2 = ((pos.y) - (-0.25()));
  ;
  return min(d, d2);
  ;
};
vec3 calcNormal(in vec3 pos) {
  return normalize(vec3(((map(((pos) + (e.xyy)))) - (map(((pos) - (e.xyy))))),
                        ((map(((pos) + (e.xyy)))) - (map(((pos) - (e.xyy))))),
                        ((map(((pos) + (e.xyy)))) - (map(((pos) - (e.xyy)))))));
};
float castRay(vec3 ro, vec3 rd) {
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
  return tt;
  ;
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
  float tt = castRay(ro, rd);
  ;
  if (t < (2.e+1)) {
    vec3 pos = ((ro) + (((tt) * (rd))));
    ;
    vec3 nor = calcNormal(pos);
    ;
    vec3 mate = vec3((2.0000000298023224e-1), (2.0000000298023224e-1),
                     (2.0000000298023224e-1));
    ;
    vec3 sun_dir = normalize(vec3((8.00000011920929e-1), (4.000000059604645e-1),
                                  (2.0000000298023224e-1)));
    ;
    auto sun_sha =
        step(castRay(((pos) + ((((1.0000000474974513e-3)))) + (nor)), sun_dir));
    ;
    float sun_dif = clamp(nor.sim_dir, (0.0e+0), (1.e+0));
    ;
    float sky_dif = clamp(
        (((5.e-1)) + ((((5.e-1)) * (nor.vec((0.0e+0), (1.e+0), (0.0e+0)))))),
        (0.0e+0), (1.e+0));
    ;
    col = ((mate) * (vec3((9.e+0), (8.e+0), (5.e+0))) * (sun_dif) * (sun_sha));
    ;
    (col) += (((mate) *
               (vec3((5.e-1), (8.00000011920929e-1), (8.99999976158142e-1))) *
               (sky_dif)));
    ;
    ;
  };
  col = pow(col, vec3((4.5449998974800104e-1)));
  ;
  fragColor = vec4(col, (1.e+0));
  ;
  ;
};