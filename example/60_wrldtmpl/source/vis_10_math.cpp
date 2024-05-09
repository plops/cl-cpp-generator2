
#include "utils.h"

#include "globals.h"

extern State state;
// implementation

float3 make_float3(float x, float y, float z) {
  nil;
  float3 f;
  (f.x) = (x);
  (f.y) = (y);
  (f.z) = (z);
  return f;
  nil;
}

float dot(float3 a, float3 b) {
  nil;
  return ((a.x) * (b.x)) + ((a.y) * (b.y)) + ((a.z) * (b.z));
  nil;
}

float rsqrtf(float f) {
  nil;
  return (1.0F) / (sqrtf(f));
  nil;
}

float3 operator*(float3 a, float s) {
  nil;
  return make_float3((s) * (a.x), (s) * (a.y), (s) * (a.z));
  nil;
}

float3 normalize(float3 v) {
  nil;
  auto invLen{rsqrtf(dot(v, v))};
  return (v) * (invLen);
  nil;
}

float3 cross(float3 a, float3 b) {
  nil;
  return make_float3(((a.y) * (b.z)) - ((a.z) * (b.y)),
                     ((a.z) * (b.x)) - ((a.x) * (b.z)),
                     ((a.x) * (b.y)) - ((a.y) * (b.x)));
  nil;
}
