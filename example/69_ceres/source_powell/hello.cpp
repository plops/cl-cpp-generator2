#include "hello_template.h"
#include <ceres/ceres.h>
#include <chrono>
#include <glog/logging.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <thread>
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  auto x1 = (3.0);
  auto x2 = (-1.0);
  auto x3 = (0.);
  auto x4 = (1.0);
  auto problem = Problem();
  problem.AddResidualBlock(new AutoDiffCostFunction<F1, 1, 1, 1>(new F1),
                           nullptr, &(x1), &(x2));
  problem.AddResidualBlock(new AutoDiffCostFunction<F2, 1, 1, 1>(new F2),
                           nullptr, &(x3), &(x4));
  problem.AddResidualBlock(new AutoDiffCostFunction<F3, 1, 1, 1>(new F3),
                           nullptr, &(x2), &(x3));
  problem.AddResidualBlock(new AutoDiffCostFunction<F4, 1, 1, 1>(new F4),
                           nullptr, &(x1), &(x4));
  auto options = Solver::Options();
  auto summary = Solver::Summary();
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::DENSE_QR;
  Solve(options, &problem, &summary);
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::setw(8)) << (" x1='") << (x1) << ("'")
                << (std::setw(8)) << (" x2='") << (x2) << ("'")
                << (std::setw(8)) << (" x3='") << (x3) << ("'")
                << (std::setw(8)) << (" x4='") << (x4) << ("'")
                << (std::setw(8)) << (" summary.BriefReport()='")
                << (summary.BriefReport()) << ("'") << (std::endl)
                << (std::flush);
  }
  return 0;
}