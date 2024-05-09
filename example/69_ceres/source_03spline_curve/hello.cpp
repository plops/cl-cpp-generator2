#include "gui.h"
#include "hello_template.h"
#include <QApplication>
#include <QMainWindow>
#include <cassert>
#include <ceres/ceres.h>
#include <chrono>
#include <cmath>
#include <glog/logging.h>
#include <iomanip>
#include <iostream>
#include <qcustomplot.h>
#include <thread>
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

int main(int argc, char **argv) {
  google::InitGoogleLogging((argv)[(0)]);
  QApplication app(argc, argv);
  MainWindow w;
  auto x0{1.0};
  auto x1{1.0};
  auto x2{1.0};
  auto x3{1.0};
  auto x4{1.0};
  auto problem{Problem()};
  auto n{67};
  auto data_x{std::vector<double>({0.,
                                   7.575757575758e-2,
                                   0.151515151515151520000000000000,
                                   0.227272727272727300000000000000,
                                   0.303030303030303040000000000000,
                                   0.378787878787878800000000000000,
                                   0.454545454545454600000000000000,
                                   0.530303030303030300000000000000,
                                   0.606060606060606100000000000000,
                                   0.681818181818181800000000000000,
                                   0.757575757575757600000000000000,
                                   0.833333333333333300000000000000,
                                   0.909090909090909200000000000000,
                                   0.984848484848484900000000000000,
                                   1.06060606060606060000000000000,
                                   1.13636363636363620000000000000,
                                   1.21212121212121220000000000000,
                                   1.28787878787878780000000000000,
                                   1.36363636363636350000000000000,
                                   1.43939393939393940000000000000,
                                   1.51515151515151510000000000000,
                                   1.59090909090909080000000000000,
                                   1.66666666666666650000000000000,
                                   1.74242424242424270000000000000,
                                   1.81818181818181830000000000000,
                                   1.89393939393939400000000000000,
                                   1.96969696969696970000000000000,
                                   2.04545454545454540000000000000,
                                   2.12121212121212100000000000000,
                                   2.19696969696969700000000000000,
                                   2.27272727272727250000000000000,
                                   2.34848484848484860000000000000,
                                   2.42424242424242430000000000000,
                                   2.50,
                                   2.57575757575757570000000000000,
                                   2.65151515151515140000000000000,
                                   2.72727272727272700000000000000,
                                   2.80303030303030280000000000000,
                                   2.87878787878787900000000000000,
                                   2.95454545454545460000000000000,
                                   3.03030303030303030000000000000,
                                   3.10606060606060600000000000000,
                                   3.18181818181818170000000000000,
                                   3.25757575757575730000000000000,
                                   3.33333333333333300000000000000,
                                   3.40909090909090870000000000000,
                                   3.48484848484848530000000000000,
                                   3.56060606060606060000000000000,
                                   3.63636363636363670000000000000,
                                   3.71212121212121200000000000000,
                                   3.78787878787878800000000000000,
                                   3.86363636363636330000000000000,
                                   3.93939393939393940000000000000,
                                   4.01515151515151500000000000000,
                                   4.09090909090909100000000000000,
                                   4.16666666666666700000000000000,
                                   4.24242424242424200000000000000,
                                   4.31818181818181800000000000000,
                                   4.39393939393939400000000000000,
                                   4.46969696969697000000000000000,
                                   4.54545454545454500000000000000,
                                   4.62121212121212100000000000000,
                                   4.69696969696969700000000000000,
                                   4.77272727272727300000000000000,
                                   4.84848484848484900000000000000,
                                   4.92424242424242400000000000000,
                                   5.0})};
  auto data_y{std::vector<double>(
      {1.13106034261593140000000000000, 1.19289290667339980000000000000,
       1.10735995156011670000000000000, 1.24850209948353960000000000000,
       1.16329309071253200000000000000, 1.17718859076735870000000000000,
       1.27803413486861370000000000000, 1.21450427291521910000000000000,
       1.40854055673020270000000000000, 1.44196393264196600000000000000,
       1.35022502007101150000000000000, 1.50730475276338030000000000000,
       1.53455507961511150000000000000, 1.57920939939709460000000000000,
       1.53930898298909870000000000000, 1.51087392968725130000000000000,
       1.65854920213765890000000000000, 1.69268555202035960000000000000,
       1.68066107553550140000000000000, 1.78582109799285900000000000000,
       1.70344566344798530000000000000, 1.69545867255466830000000000000,
       1.86177056031281670000000000000, 1.93760208288643600000000000000,
       1.87834761548492460000000000000, 1.95378358662130220000000000000,
       1.99278181211484620000000000000, 2.09829186726696000000000000000,
       2.05051886948781850000000000000, 2.10480791144796000000000000000,
       2.16786065194525000000000000000, 2.14841325262348400000000000000,
       2.29784207291344570000000000000, 2.25811542770736430000000000000,
       2.33228245281466770000000000000, 2.47783179308538330000000000000,
       2.48266370167454650000000000000, 2.58914960559537000000000000000,
       2.70129983465059600000000000000, 2.59524446569585400000000000000,
       2.81860390387209450000000000000, 2.85878683061299330000000000000,
       2.87687683350635600000000000000, 2.95473962223555600000000000000,
       2.97891521830076000000000000000, 3.16913020884417400000000000000,
       3.22210458448386300000000000000, 3.17466517005004120000000000000,
       3.27381691034599100000000000000, 3.36757619113603330000000000000,
       3.45348063972262400000000000000, 3.49409909545953030000000000000,
       3.56522451312834400000000000000, 3.65105815032873300000000000000,
       3.71834480526414040000000000000, 3.95677146582281750000000000000,
       4.02999583504663100000000000000, 4.07296362527150000000000000000,
       4.06371263030452100000000000000, 4.21405944301004700000000000000,
       4.32215324647928400000000000000, 4.42302833559313950000000000000,
       4.42500055221756500000000000000, 4.60621215495191100000000000000,
       4.78929626277662800000000000000, 4.92544985788431700000000000000,
       4.87191894063024000000000000000})};
  double params[5]{{1.0, 1.0, 1.0, 1.0, 1.0}};
  w.plot_scatter(data_x, data_y);
  for (auto i = 0; (i) < (n); (i) += (1)) {
    problem.AddResidualBlock(
        new AutoDiffCostFunction<ExponentialResidual, 1, 5>(
            new ExponentialResidual((data_x)[(i)], (data_y)[(i)])),
        nullptr, params);
  }
  auto options{Solver::Options()};
  auto summary{Solver::Summary()};
  (options.minimizer_progress_to_stdout) = (true);
  (options.max_num_iterations) = (100);
  (options.linear_solver_type) = (ceres::DENSE_QR);
  Solve(options, &problem, &summary);
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::setw(8)) << (" summary.BriefReport()='")
                << (summary.BriefReport()) << ("'") << (std::endl)
                << (std::flush);
  }
  for (auto i = 0; (i) < (5); (i) += (1)) {
    {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("") << (" ") << (std::setw(8))
                  << (" (params)[(i)]='") << ((params)[(i)]) << ("'")
                  << (std::endl) << (std::flush);
    }
  }
  auto fit_n{120};
  auto fit_x{std::vector<double>()};
  auto fit_y{std::vector<double>()};
  for (auto i = 0; (i) < (fit_n); (i) += (1)) {
    auto mi{0.};
    auto ma{5.0};
    auto x_{((ma) * (i)) / ((fit_n) - (1))};
    auto N{5};
    auto xrel{((x_) - (mi)) / ((ma) - (mi))};
    auto xpos{(xrel) * ((N) - (2))};
    auto lo_idx{int(xpos)};
    auto tau{(xpos) - (lo_idx)};
    auto hi_idx{(1) + (lo_idx)};
    assert((hi_idx) <= (4));
    assert((lo_idx) <= (3));
    assert((0) <= (hi_idx));
    assert((0) <= (lo_idx));
    auto lo_val{(params)[(lo_idx)]};
    auto hi_val{(params)[(hi_idx)]};
    auto lerp{((tau) * (lo_val)) + (((1.0) - (tau)) * ((hi_val) - (lo_val)))};
    fit_x.push_back(x_);
    fit_y.push_back(lerp);
  }
  w.plot_line(fit_x, fit_y);
  w.show();
  return app.exec();
}
