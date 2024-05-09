#pragma once
class CostFunctor {
public:
  template <typename T> bool operator()(const T *const x, T *residual) const {
    ((residual)[(0)]) = ((10.) - ((x)[(0)]));
    return true;
  }
};