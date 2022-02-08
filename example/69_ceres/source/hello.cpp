#include "hello_template.h"
#include <ceres/ceres.h>
#include <chrono>
#include <glog/logging.h>
#include <iomanip>
#include <iostream>
#include <thread>
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  auto initial_x = (5.0);
  auto x = initial_x;
  auto problem = Problem();
  auto *cost_function =
      new AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
  problem.AddResidualBlock(cost_function, nullptr, &x);
  auto options = Solver::Options();
  auto summary = Solver::Summary();
  options.minimizer_progress_to_stdout = true;
  Solve(options, &problem, &summary);
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::setw(8)) << (" initial_x='") << (initial_x)
                << ("'") << (std::setw(8)) << (" x='") << (x) << ("'")
                << (std::setw(8)) << (" summary.BriefReport()='")
                << (summary.BriefReport()) << ("'") << (std::endl)
                << (std::flush);
  }
  return 0;
}