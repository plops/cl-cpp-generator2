#ifndef VIS_08_RANDOM_H
#define VIS_08_RANDOM_H
#include "utils.h"
;
#include "globals.h"
;
// header
;

static float Noise(const int i, const int x, const int y);

static float SmoothedNoise(const int i, const int x, const int y);

static float Interpolate(const float a, const float b, const float x);

static float InterpolatedNoise(const int i, const float x, const float y);

float noise2D(const float x, const float y);

uint RandomUInt();

uint RandomUInt(uint &seed);

float RandomFloat();

float RandomFloat(&uint seed);

float Rand(float range);

float noise2D(const float x, const float y);
#endif