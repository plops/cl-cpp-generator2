float map(in vec3 pos){};
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
    (tt) += (h);
    ;
  };
  fragColor = vec4(col, (1.e+0));
  ;
  ;
};