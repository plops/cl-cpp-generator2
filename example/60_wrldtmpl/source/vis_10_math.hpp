#ifndef VIS_10_MATH_H
#define VIS_10_MATH_H
#include "utils.h"
;
#include "globals.h"
;
// header
;

float3 make_float3(float x, float y, float z);

float dot(float3 a, float3 b);

float rsqrtf(float f);

float3 operator*(float3 a, float s);

float3 normalize(float3 v);

float3 cross(float3 a, float3 b);
#endif