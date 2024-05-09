#ifndef VIS_03_MEMORY_H
#define VIS_03_MEMORY_H
#include "utils.h"
;
#include "globals.h"
;
// header

#define ALIGN(x) __attribute__((aligned(x)))
#define MALLOC64(x) ((x) == 0 ? 0 : aligned_alloc(64, (x)))
#define FREE64(x) free(x)
struct ALIGN(8) int2 {
  int x, y;
};
struct ALIGN(8) uint2 {
  uint x, y;
};
struct ALIGN(8) float2 {
  float x, y;
};
struct ALIGN(16) int3 {
  int x, y, z, dummy;
};
struct ALIGN(16) uint3 {
  uint x, y, z, dummy;
};
struct ALIGN(16) float3 {
  float x, y, z, dummy;
};
struct ALIGN(16) int4 {
  int x, y, z, w;
};
struct ALIGN(16) uint4 {
  uint x, y, z, w;
};
struct ALIGN(16) float4 {
  float x, y, z, w;
};
struct ALIGN(4) uchar4 {
  uchar x, y, z, w;
};
;
#endif