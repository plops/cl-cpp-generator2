void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  auto p =
      (((((2.e+0)) * (((fragCoord) - (iResolution.xy))))) / (iResolution.y));
  ;
  vec3 ro = vec3((0.0e+0), (0.0e+0), (2.e+0));
  ;
  vec3 rd = normalize(vec3(p, (-1.4999999999999997e+0)));
  ;
  vec3 col = vec3((0.0e+0));
  ;
  fragColor = vec4(col, (1.e+0));
  ;
  ;
};