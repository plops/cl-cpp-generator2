float sdEllipsoid(in vec3 pos) {
  float k0 = length(((pos) / (rad)));
  ;
  float k1 = length(((pos) / (((rad) * (rad)))));
  ;
  return ((((k0) * (((k0) - ((1.e+0)))))) / (k1));
  ;
};
float smin(in float a, in float b, float k) {
  float h = max(((k) - (abs(((a) - (b))))), k);
  ;
  return ((min(a, b)) - (((((h) * (h))) / (((k) * ((4.e+0)))))));
  ;
};
float sdGuy(in vec3 pos) {
  float tt = fract(iTime);
  ;
  float y = (((((4.e+0)) * (tt))) * ((((1.e+0)) - (tt))));
  ;
  float dy = (((4.e+0)) * ((((1.e+0)) - ((((2.e+0)) * (tt))))));
  ;
  vec3 u = normalize(vec3((1.e+0), dy));
  ;
  vec3 v = vec3(-dy, (1.e+0));
  ;
  vec3 cen = vec3((0.0e+0), y, (0.0e+0));
  ;
  float sy = (((5.e-1)) + ((((5.e-1)) * (y))));
  ;
  float sz = (((1.e+0)) / (sy));
  ;
  vec3 rad = vec3((2.5e-1), (((2.5e-1)) * (sy)), (((2.5e-1)) * (sz)));
  ;
  vec3 q = ((pos) - (cen));
  ;
  float d = sdEllipsoid(q, rad);
  ;
  auto h = ((pos) - (vec3((0.0e+0))));
  ;
  float d2 = sdEllipsoid(q, rad);
  ;
  return;
  ;
};
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
  auto ww = normalize(((ta) - (ro)));
  ;
  auto uu = normalize(cross(ww, vec3(0, 1, 0)));
  ;
  auto vv = normalize(cross(uu, ww));
  ;
  vec3 rd = normalize(vec3(((((p.x) * (uu))) + (((p.y) * (vv))) +
                            ((((1.4999999999999997e+0)) * (ww))))));
  ;
  vec3 col = ((vec3((6.499999761581421e-1), (7.499999999999999e-1),
                    (8.99999976158142e-1))) -
              ((((5.e-1)) * (rd.y))));
  ;
  float tt = castRay(ro, rd);
  ;
  col = mix(col,
            vec3((6.99999988079071e-1), (7.499999999999999e-1),
                 (8.00000011920929e-1)),
            exp(((-10) * (rd.y))));
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
    float bou_dif = clamp(
        (((5.e-1)) + ((((5.e-1)) * (nor.vec((0.0e+0), (-1.e+0), (0.0e+0)))))),
        (0.0e+0), (1.e+0));
    ;
    col = ((mate) * (vec3((7.e+0), (5.e+0), (3.e+0))) * (sun_dif) * (sun_sha));
    ;
    (col) += (((mate) *
               (vec3((5.e-1), (8.00000011920929e-1), (8.99999976158142e-1))) *
               (sky_dif)));
    (col) += (((mate) *
               (vec3((6.99999988079071e-1), (3.0000001192092896e-1),
                     (2.0000000298023224e-1))) *
               (bou_dif)));
    ;
    ;
  };
  col = pow(col, vec3((4.5449998974800104e-1)));
  ;
  fragColor = vec4(col, (1.e+0));
  ;
  ;
};