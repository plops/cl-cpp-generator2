#pragma once
#include <math.h>
class ExponentialResidual {
public:
  const double x_;
  const double y_;
  ExponentialResidual(double x, double y) : x_{x}, y_{y} {}
  template <typename T>
  bool operator()(const T *const m, const T *const c, T *residual) const {
    ((residual)[(0)]) = ((y_) - (exp((((m)[(0)]) * (x_)) + ((c)[(0)]))));
    return true;
  }
};