
#include "utils.h"

#include "globals.h"

extern State state;
// implementation

static uint seed = 0x12345678;
static int numX = 512, numY = 512, numOctaves = 7, primeIndex = 0;
static float persistence = .5f;
static int primes[10][3]{{{995615039, 600173719, 701464987},
                          {831731269, 162318869, 136250887},
                          {174329291, 946737083, 245679977},
                          {362489573, 795918041, 350777237},
                          {457025711, 880830799, 909678923},
                          {787070341, 177340217, 593320781},
                          {405493717, 291031019, 391950901},
                          {458904767, 676625681, 424452397},
                          {531736441, 939683957, 810651871},
                          {997169939, 842027887, 423882827}}};

static float Noise(const int i, const int x, const int y) {
  nil;
  auto n{(x) + ((57) * (y))};
  (n) = (((n) << (13)) ^ (n));
  auto a{(primes)[(i)][(0)]};
  auto b{(primes)[(i)][(1)]};
  auto c{(primes)[(i)][(2)]};
  auto tt{((((n) * (((n) * (n) * (a)) + (b))) + (c)) & (2147483647))};
  return (1.0F) - ((static_cast<float>(tt)) / (1.1e+9F));
  nil;
}

static float SmoothedNoise(const int i, const int x, const int y) {
  nil;
  auto corners{((Noise(i, x - 1, y - 1)) + (Noise(i, x + 1, y - 1)) +
                (Noise(i, x - 1, y + 1)) + (Noise(i, x + 1, y + 1))) /
               (16)};
  auto sides{((Noise(i, x - 1, y)) + (Noise(i, x + 1, y)) +
              (Noise(i, x, y - 1)) + (Noise(i, x, y + 1))) /
             (8)};
  auto center{(Noise(i, x, y)) / (4)};
  return (corners) + (sides) + (center);
  nil;
}

static float Interpolate(const float a, const float b, const float x) {
  nil;
  auto ft{(x) * (3.141593F)};
  auto f{(0.50F) * ((1.0F) - (cosf(ft)))};
  return ((a) * ((1) - (f))) + ((b) * (f));
  nil;
}

static float InterpolatedNoise(const int i, const float x, const float y) {
  nil;
  auto integer_X{static_cast<int>(x)};
  auto integer_Y{static_cast<int>(y)};
  auto fractional_X{(x) - (integer_X)};
  auto fractional_Y{(y) - (integer_Y)};
  auto v0{SmoothedNoise(i, integer_X, integer_Y)};
  auto v1{SmoothedNoise(i, integer_X + 1, integer_Y)};
  auto v2{SmoothedNoise(i, integer_X, integer_Y + 1)};
  auto v3{SmoothedNoise(i, integer_X + 1, integer_Y + 1)};
  auto i1{Interpolate(v1, v2, fractional_X)};
  auto i2{Interpolate(v3, v4, fractional_X)};
  return Interpolate(i1, i2, fractional_Y);
  nil;
}

float noise2D(const float x, const float y) {
  nil;
  auto total{0.F};
  auto frequency{static_cast<float>((2) << (numOctaves))};
  auto amplitude{1.0F};
  for (auto i = 0; (i) < (numOctaves); (i) += (1)) {
    (frequency) /= (2);
    (amplitude) *= (persistance);
    (total) += ((amplitude) *
                (InterpolatedNoise(((primeIndex) + (1)) % (10),
                                   (x) / (frequency), (y) / (frequency))));
  }
  return (total) / (frequency);
  nil;
}

uint RandomUInt() {
  nil;
  (seed) ^= ((seed) << (13));
  (seed) ^= ((seed) >> (17));
  (seed) ^= ((seed) << (5));
  return seed;
  nil;
}

uint RandomUInt(uint &seed) {
  nil;
  (seed) ^= ((seed) << (13));
  (seed) ^= ((seed) >> (17));
  (seed) ^= ((seed) << (5));
  return seed;
  nil;
}

float RandomFloat() {
  nil;
  return (RandomUInt()) * (2.3283064365387e-10f);
  nil;
}

float RandomFloat(&uint seed) {
  nil;
  return (RandomUInt(seed)) * (2.3283064365387e-10f);
  nil;
}

float Rand(float range) {
  nil;
  return (RandomFloat()) * (range);
  nil;
}

float noise2D(const float x, const float y) {
  nil;
  nil;
  nil;
}
