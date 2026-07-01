// --- transpiled raymarching shader from declarative DSL ---

float smin(float a, float b, float k) {
  float h;
  h = clamp(0.50F + 0.50F * ((b - a) / k), 0.F, 1.0F);
  return mix(b, a, h) - (k * h * (1.0F - h));
}

vec3 rotateX(vec3 p, float a) {
  float c;
  float s;
  c = cos(a);
  s = sin(a);
  return vec3(p.x, (c * p.y) - (s * p.z), s * p.y + c * p.z);
}

vec3 rotateY(vec3 p, float a) {
  float c;
  float s;
  c = cos(a);
  s = sin(a);
  return vec3(c * p.x + s * p.z, p.y, (c * p.z) - (s * p.x));
}

vec3 rotateZ(vec3 p, float a) {
  float c;
  float s;
  c = cos(a);
  s = sin(a);
  return vec3((c * p.x) - (s * p.y), s * p.x + c * p.y, p.z);
}

float simple_noise(vec3 p) {
  return 0.3330F * (sin(p.x) + sin(p.y) + sin(p.z));
}

float sdSphere(vec3 p, float s) { return length(p) - s; }

float sdBox(vec3 p, vec3 b) {
  vec3 q;
  q = (abs(p) - b);
  return length(max(q, 0.F)) + min(max(q.x, max(q.y, q.z)), 0.F);
}

float sdTorus(vec3 p, vec2 tVal) {
  vec2 q;
  q = vec2(length(p.xz) - (tVal.x), p.y);
  return length(q) - (tVal.y);
}

float sdCylinder(vec3 p, vec2 h) {
  vec2 d;
  d = (abs(vec2(length(p.xz), p.y)) - h);
  return min(max(d.x, d.y), 0.F) + length(max(d, 0.F));
}

float map(vec3 p, float heat_intensity, float spin_speed, float melt_factor) {
  return min(
      p.y + 1.50F,
      max(-sdCylinder(p, vec2(0.350F, 3.0F)),
          smin(sdSphere(p, 0.850F + 0.10F * sin(iTime * 3.0F)) +
                   heat_intensity * 0.150F *
                       simple_noise(p * 3.0F + vec3(0.F, iTime * 2.0F, 0.F)),
               min(sdTorus(rotateX(p, iTime * spin_speed), vec2(1.10F, 0.120F)),
                   sdTorus(rotateZ(p, iTime * spin_speed * 0.80F),
                           vec2(1.250F, 0.10F))),
               melt_factor)));
}

vec3 getNormal(vec3 p, float heat_intensity, float spin_speed,
               float melt_factor) {
  vec2 e;
  float d;
  vec3 n;
  e = vec2(1.00e-3F, 0.F);
  d = map(p, heat_intensity, spin_speed, melt_factor);
  n = (d - vec3(map(p - e.xyy, heat_intensity, spin_speed, melt_factor),
                map(p - e.yxy, heat_intensity, spin_speed, melt_factor),
                map(p - e.yyx, heat_intensity, spin_speed, melt_factor)));
  return normalize(n);
}

float getShadow(vec3 ro, vec3 rd, float mint, float maxt, float k,
                float heat_intensity, float spin_speed, float melt_factor) {
  float res;
  float tVal;
  res = 1.0F;
  tVal = mint;
  for (int i = 0; i < 32; i++) {
    float h;
    h = map(ro + tVal * rd, heat_intensity, spin_speed, melt_factor);
    if (h < 1.00e-3F) {
      return 0.F;
    }
    res = min(res, (k * h) / tVal);
    tVal += clamp(h, 1.00e-2F, 0.20F);
    if (tVal > maxt) {
      break;
    }
  }
  return clamp(res, 0.F, 1.0F);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  vec4 state;
  state = texelFetch(iChannel0, ivec2(0, 0), 0);
  float heat_intensity;
  float spin_speed;
  float focused_widget;
  float melt_factor;
  heat_intensity = state.x;
  spin_speed = state.y;
  focused_widget = state.z;
  melt_factor = state.w;
  vec2 uv;
  vec3 ro;
  vec3 rd;
  float tVal;
  bool hit;
  vec3 p;
  vec3 n;
  vec3 lightPos;
  vec3 l;
  float dif;
  float shadow;
  vec3 objectColor;
  vec3 col;
  uv = ((fragCoord - (0.50F * iResolution.xy)) / iResolution.y);
  ro = vec3(0.F, 1.0F, -4.50F);
  rd = normalize(vec3(uv, 1.0F));
  tVal = 0.F;
  hit = false;
  for (int i = 0; i < 80; i++) {
    float d;
    d = map(ro + tVal * rd, heat_intensity, spin_speed, melt_factor);
    if (d < 1.00e-3F) {
      hit = true;
      break;
    }
    tVal += d;
    if (tVal > 12.F) {
      break;
    }
  }
  col = vec3(2.00e-2F, 2.00e-2F, 4.00e-2F);
  if (hit) {
    p = ro + tVal * rd;
    n = getNormal(p, heat_intensity, spin_speed, melt_factor);
    lightPos = vec3(2.0F, 4.0F, -3.0F);
    l = normalize(lightPos - p);
    dif = clamp(dot(n, l), 0.F, 1.0F);
    shadow = getShadow(p + n * 1.00e-2F, l, 1.00e-2F, 5.0F, 16.F,
                       heat_intensity, spin_speed, melt_factor);
    if (p.y > -1.490F) {
      float centerDist;
      centerDist = length(p);
      objectColor = mix(vec3(0.90F, 0.30F, 0.F), vec3(0.70F, 0.70F, 0.80F),
                        clamp(centerDist, 0.F, 1.0F));
    } else {
      objectColor = vec3(0.150F, 0.150F, 0.150F);
    }
    col = objectColor * (dif * shadow + 0.10F);
    col = pow(col, vec3(0.45450F));
  }
  vec2 scr_uv;
  vec3 bar_color;
  vec3 handle_color;
  vec3 focus_color;
  scr_uv = (fragCoord / iResolution.xy);
  bar_color = vec3(0.30F);
  handle_color = vec3(0.70F);
  focus_color = vec3(0.90F, 0.30F, 0.10F);
  float VAL_0;
  float Y_CENTER_0;
  bool IS_FOCUSED_0;
  VAL_0 = ((heat_intensity - 0.F) / 2.0F);
  Y_CENTER_0 = 0.1250F;
  IS_FOCUSED_0 = focused_widget == 0.F;
  float HX_0;
  HX_0 = 5.00e-2F + VAL_0 * 0.350F;
  if (scr_uv.x >= 5.00e-2F && scr_uv.x <= 0.40F &&
      abs(scr_uv.y - Y_CENTER_0) < 6.00e-3F) {
    col = mix(col, IS_FOCUSED_0 ? focus_color : bar_color, 0.80F);
  }
  if (length(scr_uv - vec2(HX_0, Y_CENTER_0)) < 1.20e-2F) {
    col = mix(col, IS_FOCUSED_0 ? focus_color : handle_color, 1.0F);
  }
  float VAL_1;
  float Y_CENTER_1;
  bool IS_FOCUSED_1;
  VAL_1 = ((spin_speed - 0.F) / 4.0F);
  Y_CENTER_1 = 0.2050F;
  IS_FOCUSED_1 = focused_widget == 1.0F;
  float HX_1;
  HX_1 = 5.00e-2F + VAL_1 * 0.350F;
  if (scr_uv.x >= 5.00e-2F && scr_uv.x <= 0.40F &&
      abs(scr_uv.y - Y_CENTER_1) < 6.00e-3F) {
    col = mix(col, IS_FOCUSED_1 ? focus_color : bar_color, 0.80F);
  }
  if (length(scr_uv - vec2(HX_1, Y_CENTER_1)) < 1.20e-2F) {
    col = mix(col, IS_FOCUSED_1 ? focus_color : handle_color, 1.0F);
  }
  float VAL_2;
  float Y_CENTER_2;
  bool IS_FOCUSED_2;
  VAL_2 = ((melt_factor - 0.10F) / 1.40F);
  Y_CENTER_2 = 0.2850F;
  IS_FOCUSED_2 = focused_widget == 2.0F;
  float HX_2;
  HX_2 = 5.00e-2F + VAL_2 * 0.350F;
  if (scr_uv.x >= 5.00e-2F && scr_uv.x <= 0.40F &&
      abs(scr_uv.y - Y_CENTER_2) < 6.00e-3F) {
    col = mix(col, IS_FOCUSED_2 ? focus_color : bar_color, 0.80F);
  }
  if (length(scr_uv - vec2(HX_2, Y_CENTER_2)) < 1.20e-2F) {
    col = mix(col, IS_FOCUSED_2 ? focus_color : handle_color, 1.0F);
  }
  fragColor = vec4(col, 1.0F);
}
