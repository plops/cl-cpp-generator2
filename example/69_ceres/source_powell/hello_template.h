#pragma once
#include <math.h>
class F1 {
public:
  template <typename T>
  bool operator()(const T *const x1, const T *const x2, T *residual) const {
    residual[0] = ((x1[0]) + ((((10.)) * (x2[0]))));
    return true;
  }
};
class F2 {
public:
  template <typename T>
  bool operator()(const T *const x3, const T *const x4, T *residual) const {
    residual[0] = ((sqrt((5.0))) * (((x3[0]) - (x4[0]))));
    return true;
  }
};
class F3 {
public:
  template <typename T>
  bool operator()(const T *const x2, const T *const x3, T *residual) const {
    residual[0] = ((((x2[0]) - ((((2.0)) * (x3[0]))))) *
                   (((x2[0]) - ((((2.0)) * (x3[0]))))));
    return true;
  }
};
class F4 {
public:
  template <typename T>
  bool operator()(const T *const x1, const T *const x4, T *residual) const {
    residual[0] =
        ((sqrt((10.))) * (((x1[0]) - (x4[0]))) * (((x1[0]) - (x4[0]))));
    return true;
  }
};