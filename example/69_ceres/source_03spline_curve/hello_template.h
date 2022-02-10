#pragma once
#include <ceres/ceres.h>
#include <cmath>
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
class ExponentialResidual {
public:
  const double x_;
  const double y_;
  ExponentialResidual(double x, double y) : x_(x), y_(y) {}
  template <typename T> bool operator()(const T *const x0, T *residual) const {
    // piecewise linear interpolation of equidistant points
    ;
    auto mi = (0.);
    auto ma = (5.0);
    auto N = 5;
    auto xrel = ((((x_) - (mi))) / (((ma) - (mi))));
    auto xpos = ((xrel) * (((N) - (2))));
    auto lo_idx = int(xpos);
    auto tau = ((xpos) - (lo_idx));
    auto hi_idx = ((1) + (lo_idx));
    auto lo_val = x0[lo_idx];
    auto hi_val = x0[hi_idx];
    auto lerp = ((((tau) * (lo_val))) +
                 ((((((1.0)) - (tau))) * (((hi_val) - (lo_val))))));
    residual[0] = ((y_) - (lerp));
    return true;
  }
};