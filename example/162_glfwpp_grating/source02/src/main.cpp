#include "/home/martin/src/popl/include/popl.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <format>
#include <glfwpp/glfwpp.h>
#include <iostream>
#include <string>
#include <thread>
#include <valarray>
#include <vector>
using namespace std;
using namespace chrono;
using Scalar = float;
using Vec = std::vector<Scalar>;
using VecI = const Vec;
class Statistics {
public:
  int getSignificantDigits(Scalar num) {
    if (num == 0.F) {
      return 1;
    }
    if (num < 0) {
      num = -num;
    }
    auto significantDigits{0};
    while (num <= 1.0F) {
      num *= 10.F;
      significantDigits++;
    }
    return significantDigits;
  }
  string printStat(tuple<Scalar, Scalar, Scalar, Scalar> m_md_d_dd) {
    auto [m, md, d, dd]{m_md_d_dd};
    const auto rel{1.00e+2F * (d / m)};
    const auto mprecision{getSignificantDigits(md)};
    const auto dprecision{getSignificantDigits(dd)};
    const auto rprecision{getSignificantDigits(rel)};
    const auto fmtm{std::string("{:.") + to_string(mprecision) + "f}"};
    const auto fmtd{std::string("{:.") + to_string(dprecision) + "f}"};
    const auto fmtr{std::string(" ({:.") + to_string(rprecision) + "f}%)"};
    const auto format_str{fmtm + "±" + fmtd + fmtr};
    return vformat(format_str, make_format_args(m, d, rel));
  }
  Statistics(int n) : numberFramesForStatistics{n}, fitres{deque<float>()} {}
  deque<float> fitres;
  int numberFramesForStatistics;
  void push_back(float frameTimems) {
    fitres.push_back(frameTimems);
    if (numberFramesForStatistics < fitres.size()) {
      fitres.pop_front();
    }
  }
  tuple<Scalar, Scalar, Scalar, Scalar> compute() {
    auto computeStat{[](const auto &fitres,
                        auto filter) -> tuple<Scalar, Scalar, Scalar, Scalar> {
      // compute mean and standard deviation Numerical Recipes 14.1.2 and 14.1.8
      auto data{valarray<Scalar>(fitres.size())};
      data.resize(fitres.size());
      transform(fitres.begin(), fitres.end(), &data[0], filter);
      const auto N{static_cast<Scalar>(data.size())};
      const auto mean{(data.sum()) / N};
      const auto stdev{sqrt(
          ((pow(data - mean, 2).sum()) - (pow((data - mean).sum(), 2) / N)) /
          (N - 1.0F))};
      const auto mean_stdev{stdev / sqrt(N)};
      const auto stdev_stdev{stdev / sqrt(2 * N)};
      return make_tuple(mean, mean_stdev, stdev, stdev_stdev);
    }};
    return computeStat(fitres, [&](const auto &f) { return f; });
  }
};
class DelayEstimator {
public:
  DelayEstimator(int numberFramesForStatistics)
      : numberFramesForStatistics{numberFramesForStatistics},
        frameRateStats{Statistics(numberFramesForStatistics)} {
    t0 = high_resolution_clock::now();
  }
  int numberFramesForStatistics;
  Statistics frameRateStats;
  decltype(high_resolution_clock::now()) t0;
  decltype(high_resolution_clock::now()) t1;
  void update() {
    t1 = high_resolution_clock::now();
    auto frameTimens{duration_cast<nanoseconds>(t1 - t0).count()};
    auto frameTimems{frameTimens / 1.0e+6F};
    auto frameRateHz{1.0e+9F / frameTimens};
    frameRateStats.push_back(frameTimems);
    const auto cs{frameRateStats.compute()};
    const auto pcs{frameRateStats.printStat(cs)};
    auto [frameTime_, frameTime_Std, frameTimeStd, frameTimeStdStd]{cs};
    std::cout << std::format(
        "(:pcs '{}' :frameTimems '{}' :frameRateHz '{}')\n", pcs, frameTimems,
        frameRateHz);
    t0 = t1;
  }
};

int main(int argc, char **argv) {
  auto op{popl::OptionParser("allowed options")};
  auto swapInterval{int(2)};
  auto numberFramesForStatistics{int(211)};
  auto darkLevel{int(0)};
  auto brightLevel{int(255)};
  auto helpOption{op.add<popl::Switch>("h", "help", "produce help message")};
  auto verboseOption{
      op.add<popl::Switch>("v", "verbose", "produce verbose output")};
  auto swapIntervalOption{op.add<popl::Value<int>>(
      "s", "swapInterval", "parameter", 2, &swapInterval)};
  auto numberFramesForStatisticsOption{
      op.add<popl::Value<int>>("F", "numberFramesForStatistics", "parameter",
                               211, &numberFramesForStatistics)};
  auto darkLevelOption{
      op.add<popl::Value<int>>("D", "darkLevel", "parameter", 0, &darkLevel)};
  auto brightLevelOption{op.add<popl::Value<int>>(
      "B", "brightLevel", "parameter", 255, &brightLevel)};
  op.parse(argc, argv);
  if (helpOption->is_set()) {
    cout << op << endl;
    exit(0);
  }
  auto frameDelayEstimator{DelayEstimator(numberFramesForStatistics)};
  auto dark{darkLevel / 255.F};
  auto bright{brightLevel / 255.F};
  class DrawPrimitive {
  public:
    array<float, 3> color;
    decltype(GL_QUADS) type;
    vector<array<float, 4>> coords;
  };
  class DrawFrame {
  public:
    int id;
    string name;
    vector<DrawPrimitive> draw;

    void execute() {
      for (auto &&[color, type, coords] : draw) {
        glColor4f(color[0], color[1], color[2], 1.0F);
        glBegin(type);
        for (auto &&[x0, y0, x1, y1] : coords) {
          glVertex2f(x0, y0);
          glVertex2f(x1, y0);
          glVertex2f(x1, y1);
          glVertex2f(x0, y1);
        }
        glEnd();
      }
    }
  };
  auto w{512};
  auto h{512};
  auto wf{static_cast<float>(w)};
  auto hf{static_cast<float>(h)};
  // show a sequence of horizontal bars and vertical bars that split the image
  // into 1/2, 1/4th, ... . each image is followed by its inverted version. the
  // lcd of the projector is too slow to show this pattern exactly with 60Hz.
  // that is why we set swap interval to 2 (we wait for two frames for every
  // image so that the display has time to settle)
  vector<DrawFrame> drawFrames =
      {{.id = 0,
        .name = "all-white",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}}}},
       {.id = 1,
        .name = "all-dark",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}}}},
       {.id = 2,
        .name = "vertical-stripes-0-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 512.F, hf}}}}},
       {.id = 3,
        .name = "vertical-stripes-0-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 512.F, hf}}}}},
       {.id = 4,
        .name = "vertical-stripes-1-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 256.F, hf}, {512.F, 0.F, 768.F, hf}}}}},
       {.id = 5,
        .name = "vertical-stripes-1-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 256.F, hf}, {512.F, 0.F, 768.F, hf}}}}},
       {.id = 6,
        .name = "vertical-stripes-2-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 128.F, hf},
                             {256.F, 0.F, 384.F, hf},
                             {512.F, 0.F, 6.40e+2F, hf},
                             {768.F, 0.F, 896.F, hf}}}}},
       {.id = 7,
        .name = "vertical-stripes-2-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 128.F, hf},
                             {256.F, 0.F, 384.F, hf},
                             {512.F, 0.F, 6.40e+2F, hf},
                             {768.F, 0.F, 896.F, hf}}}}},
       {.id = 8,
        .name = "vertical-stripes-3-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 64.F, hf},
                             {128.F, 0.F, 192.F, hf},
                             {256.F, 0.F, 3.20e+2F, hf},
                             {384.F, 0.F, 448.F, hf},
                             {512.F, 0.F, 576.F, hf},
                             {6.40e+2F, 0.F, 704.F, hf},
                             {768.F, 0.F, 832.F, hf},
                             {896.F, 0.F, 9.60e+2F, hf}}}}},
       {.id = 9,
        .name = "vertical-stripes-3-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 64.F, hf},
                             {128.F, 0.F, 192.F, hf},
                             {256.F, 0.F, 3.20e+2F, hf},
                             {384.F, 0.F, 448.F, hf},
                             {512.F, 0.F, 576.F, hf},
                             {6.40e+2F, 0.F, 704.F, hf},
                             {768.F, 0.F, 832.F, hf},
                             {896.F, 0.F, 9.60e+2F, hf}}}}},
       {.id = 10,
        .name = "vertical-stripes-4-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 32.F, hf},
                             {64.F, 0.F, 96.F, hf},
                             {128.F, 0.F, 1.60e+2F, hf},
                             {192.F, 0.F, 224.F, hf},
                             {256.F, 0.F, 288.F, hf},
                             {3.20e+2F, 0.F, 352.F, hf},
                             {384.F, 0.F, 416.F, hf},
                             {448.F, 0.F, 4.80e+2F, hf},
                             {512.F, 0.F, 544.F, hf},
                             {576.F, 0.F, 608.F, hf},
                             {6.40e+2F, 0.F, 672.F, hf},
                             {704.F, 0.F, 736.F, hf},
                             {768.F, 0.F, 8.00e+2F, hf},
                             {832.F, 0.F, 864.F, hf},
                             {896.F, 0.F, 928.F, hf},
                             {9.60e+2F, 0.F, 992.F, hf}}}}},
       {.id = 11,
        .name = "vertical-stripes-4-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 32.F, hf},
                             {64.F, 0.F, 96.F, hf},
                             {128.F, 0.F, 1.60e+2F, hf},
                             {192.F, 0.F, 224.F, hf},
                             {256.F, 0.F, 288.F, hf},
                             {3.20e+2F, 0.F, 352.F, hf},
                             {384.F, 0.F, 416.F, hf},
                             {448.F, 0.F, 4.80e+2F, hf},
                             {512.F, 0.F, 544.F, hf},
                             {576.F, 0.F, 608.F, hf},
                             {6.40e+2F, 0.F, 672.F, hf},
                             {704.F, 0.F, 736.F, hf},
                             {768.F, 0.F, 8.00e+2F, hf},
                             {832.F, 0.F, 864.F, hf},
                             {896.F, 0.F, 928.F, hf},
                             {9.60e+2F, 0.F, 992.F, hf}}}}},
       {.id = 12,
        .name = "vertical-stripes-5-normal",
        .draw =
            {{.color = {dark, dark, dark},
              .type =
                  GL_QUADS,
              .coords = {{0.F, 0.F, wf, hf}}},
             {.color = {bright, bright, bright},
              .type =
                  GL_QUADS,
              .coords =
                  {{0.F, 0.F, 16.F, hf},       {32.F, 0.F, 48.F, hf},
                   {64.F, 0.F, 80.F, hf},      {96.F, 0.F, 112.F, hf},
                   {128.F, 0.F, 144.F, hf},    {1.60e+2F, 0.F, 176.F, hf},
                   {192.F, 0.F, 208.F, hf},    {224.F, 0.F, 2.40e+2F, hf},
                   {256.F, 0.F, 272.F, hf},    {288.F, 0.F, 304.F, hf},
                   {3.20e+2F, 0.F, 336.F, hf}, {352.F, 0.F, 368.F, hf},
                   {384.F, 0.F, 4.00e+2F, hf}, {416.F, 0.F, 432.F, hf},
                   {448.F, 0.F, 464.F, hf},    {4.80e+2F, 0.F, 496.F, hf},
                   {512.F, 0.F, 528.F, hf},    {544.F, 0.F, 5.60e+2F, hf},
                   {576.F, 0.F, 592.F, hf},    {608.F, 0.F, 624.F, hf},
                   {6.40e+2F, 0.F, 656.F, hf}, {672.F, 0.F, 688.F, hf},
                   {704.F, 0.F, 7.20e+2F, hf}, {736.F, 0.F, 752.F, hf},
                   {768.F, 0.F, 784.F, hf},    {8.00e+2F, 0.F, 816.F, hf},
                   {832.F, 0.F, 848.F, hf},    {864.F, 0.F, 8.80e+2F, hf},
                   {896.F, 0.F, 912.F, hf},    {928.F, 0.F, 944.F, hf},
                   {9.60e+2F, 0.F, 976.F, hf}, {992.F, 0.F, 1008.F, hf}}}}},
       {.id = 13,
        .name = "vertical-stripes-5-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords =
                      {{0.F, 0.F, 16.F, hf},       {32.F, 0.F, 48.F, hf},
                       {64.F, 0.F, 80.F, hf},      {96.F, 0.F, 112.F, hf},
                       {128.F, 0.F, 144.F, hf},    {1.60e+2F, 0.F, 176.F, hf},
                       {192.F, 0.F, 208.F, hf},    {224.F, 0.F, 2.40e+2F, hf},
                       {256.F, 0.F, 272.F, hf},    {288.F, 0.F, 304.F, hf},
                       {3.20e+2F, 0.F, 336.F, hf}, {352.F, 0.F, 368.F, hf},
                       {384.F, 0.F, 4.00e+2F, hf}, {416.F, 0.F, 432.F, hf},
                       {448.F, 0.F, 464.F, hf},    {4.80e+2F, 0.F, 496.F, hf},
                       {512.F, 0.F, 528.F, hf},    {544.F, 0.F, 5.60e+2F, hf},
                       {576.F, 0.F, 592.F, hf},    {608.F, 0.F, 624.F, hf},
                       {6.40e+2F, 0.F, 656.F, hf}, {672.F, 0.F, 688.F, hf},
                       {704.F, 0.F, 7.20e+2F, hf}, {736.F, 0.F, 752.F, hf},
                       {768.F, 0.F, 784.F, hf},    {8.00e+2F, 0.F, 816.F, hf},
                       {832.F, 0.F, 848.F, hf},    {864.F, 0.F, 8.80e+2F, hf},
                       {896.F, 0.F, 912.F, hf},    {928.F, 0.F, 944.F, hf},
                       {9.60e+2F, 0.F, 976.F, hf}, {992.F, 0.F, 1008.F, hf}}}}},
       {.id = 14,
        .name = "vertical-stripes-6-normal",
        .draw =
            {{.color = {dark, dark, dark},
              .type =
                  GL_QUADS,
              .coords = {{0.F, 0.F, wf, hf}}},
             {.color = {bright, bright, bright},
              .type =
                  GL_QUADS,
              .coords =
                  {{0.F, 0.F, 8.0F, hf},       {16.F, 0.F, 24.F, hf},
                   {32.F, 0.F, 40.F, hf},      {48.F, 0.F, 56.F, hf},
                   {64.F, 0.F, 72.F, hf},      {80.F, 0.F, 88.F, hf},
                   {96.F, 0.F, 104.F, hf},     {112.F, 0.F, 1.20e+2F, hf},
                   {128.F, 0.F, 136.F, hf},    {144.F, 0.F, 152.F, hf},
                   {1.60e+2F, 0.F, 168.F, hf}, {176.F, 0.F, 184.F, hf},
                   {192.F, 0.F, 2.00e+2F, hf}, {208.F, 0.F, 216.F, hf},
                   {224.F, 0.F, 232.F, hf},    {2.40e+2F, 0.F, 248.F, hf},
                   {256.F, 0.F, 264.F, hf},    {272.F, 0.F, 2.80e+2F, hf},
                   {288.F, 0.F, 296.F, hf},    {304.F, 0.F, 312.F, hf},
                   {3.20e+2F, 0.F, 328.F, hf}, {336.F, 0.F, 344.F, hf},
                   {352.F, 0.F, 3.60e+2F, hf}, {368.F, 0.F, 376.F, hf},
                   {384.F, 0.F, 392.F, hf},    {4.00e+2F, 0.F, 408.F, hf},
                   {416.F, 0.F, 424.F, hf},    {432.F, 0.F, 4.40e+2F, hf},
                   {448.F, 0.F, 456.F, hf},    {464.F, 0.F, 472.F, hf},
                   {4.80e+2F, 0.F, 488.F, hf}, {496.F, 0.F, 504.F, hf},
                   {512.F, 0.F, 5.20e+2F, hf}, {528.F, 0.F, 536.F, hf},
                   {544.F, 0.F, 552.F, hf},    {5.60e+2F, 0.F, 568.F, hf},
                   {576.F, 0.F, 584.F, hf},    {592.F, 0.F, 6.00e+2F, hf},
                   {608.F, 0.F, 616.F, hf},    {624.F, 0.F, 632.F, hf},
                   {6.40e+2F, 0.F, 648.F, hf}, {656.F, 0.F, 664.F, hf},
                   {672.F, 0.F, 6.80e+2F, hf}, {688.F, 0.F, 696.F, hf},
                   {704.F, 0.F, 712.F, hf},    {7.20e+2F, 0.F, 728.F, hf},
                   {736.F, 0.F, 744.F, hf},    {752.F, 0.F, 7.60e+2F, hf},
                   {768.F, 0.F, 776.F, hf},    {784.F, 0.F, 792.F, hf},
                   {8.00e+2F, 0.F, 808.F, hf}, {816.F, 0.F, 824.F, hf},
                   {832.F, 0.F, 8.40e+2F, hf}, {848.F, 0.F, 856.F, hf},
                   {864.F, 0.F, 872.F, hf},    {8.80e+2F, 0.F, 888.F, hf},
                   {896.F, 0.F, 904.F, hf},    {912.F, 0.F, 9.20e+2F, hf},
                   {928.F, 0.F, 936.F, hf},    {944.F, 0.F, 952.F, hf},
                   {9.60e+2F, 0.F, 968.F, hf}, {976.F, 0.F, 984.F, hf},
                   {992.F, 0.F, 1.00e+3F, hf}, {1008.F, 0.F, 1016.F, hf}}}}},
       {.id = 15,
        .name = "vertical-stripes-6-invert",
        .draw =
            {{.color = {bright, bright, bright},
              .type =
                  GL_QUADS,
              .coords = {{0.F, 0.F, wf, hf}}},
             {.color = {dark, dark, dark},
              .type =
                  GL_QUADS,
              .coords =
                  {{0.F, 0.F, 8.0F, hf},       {16.F, 0.F, 24.F, hf},
                   {32.F, 0.F, 40.F, hf},      {48.F, 0.F, 56.F, hf},
                   {64.F, 0.F, 72.F, hf},      {80.F, 0.F, 88.F, hf},
                   {96.F, 0.F, 104.F, hf},     {112.F, 0.F, 1.20e+2F, hf},
                   {128.F, 0.F, 136.F, hf},    {144.F, 0.F, 152.F, hf},
                   {1.60e+2F, 0.F, 168.F, hf}, {176.F, 0.F, 184.F, hf},
                   {192.F, 0.F, 2.00e+2F, hf}, {208.F, 0.F, 216.F, hf},
                   {224.F, 0.F, 232.F, hf},    {2.40e+2F, 0.F, 248.F, hf},
                   {256.F, 0.F, 264.F, hf},    {272.F, 0.F, 2.80e+2F, hf},
                   {288.F, 0.F, 296.F, hf},    {304.F, 0.F, 312.F, hf},
                   {3.20e+2F, 0.F, 328.F, hf}, {336.F, 0.F, 344.F, hf},
                   {352.F, 0.F, 3.60e+2F, hf}, {368.F, 0.F, 376.F, hf},
                   {384.F, 0.F, 392.F, hf},    {4.00e+2F, 0.F, 408.F, hf},
                   {416.F, 0.F, 424.F, hf},    {432.F, 0.F, 4.40e+2F, hf},
                   {448.F, 0.F, 456.F, hf},    {464.F, 0.F, 472.F, hf},
                   {4.80e+2F, 0.F, 488.F, hf}, {496.F, 0.F, 504.F, hf},
                   {512.F, 0.F, 5.20e+2F, hf}, {528.F, 0.F, 536.F, hf},
                   {544.F, 0.F, 552.F, hf},    {5.60e+2F, 0.F, 568.F, hf},
                   {576.F, 0.F, 584.F, hf},    {592.F, 0.F, 6.00e+2F, hf},
                   {608.F, 0.F, 616.F, hf},    {624.F, 0.F, 632.F, hf},
                   {6.40e+2F, 0.F, 648.F, hf}, {656.F, 0.F, 664.F, hf},
                   {672.F, 0.F, 6.80e+2F, hf}, {688.F, 0.F, 696.F, hf},
                   {704.F, 0.F, 712.F, hf},    {7.20e+2F, 0.F, 728.F, hf},
                   {736.F, 0.F, 744.F, hf},    {752.F, 0.F, 7.60e+2F, hf},
                   {768.F, 0.F, 776.F, hf},    {784.F, 0.F, 792.F, hf},
                   {8.00e+2F, 0.F, 808.F, hf}, {816.F, 0.F, 824.F, hf},
                   {832.F, 0.F, 8.40e+2F, hf}, {848.F, 0.F, 856.F, hf},
                   {864.F, 0.F, 872.F, hf},    {8.80e+2F, 0.F, 888.F, hf},
                   {896.F, 0.F, 904.F, hf},    {912.F, 0.F, 9.20e+2F, hf},
                   {928.F, 0.F, 936.F, hf},    {944.F, 0.F, 952.F, hf},
                   {9.60e+2F, 0.F, 968.F, hf}, {976.F, 0.F, 984.F, hf},
                   {992.F, 0.F, 1.00e+3F, hf}, {1008.F, 0.F, 1016.F, hf}}}}},
       {.id = 16,
        .name = "horizontal-stripes-0-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 512.F, hf}}}}},
       {.id = 17,
        .name = "horizontal-stripes-0-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 512.F, hf}}}}},
       {.id = 18,
        .name = "horizontal-stripes-1-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 256.F, hf}, {512.F, 0.F, 768.F, hf}}}}},
       {.id = 19,
        .name = "horizontal-stripes-1-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 256.F, hf}, {512.F, 0.F, 768.F, hf}}}}},
       {.id = 20,
        .name = "horizontal-stripes-2-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 128.F, hf},
                             {256.F, 0.F, 384.F, hf},
                             {512.F, 0.F, 6.40e+2F, hf},
                             {768.F, 0.F, 896.F, hf}}}}},
       {.id = 21,
        .name = "horizontal-stripes-2-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 128.F, hf},
                             {256.F, 0.F, 384.F, hf},
                             {512.F, 0.F, 6.40e+2F, hf},
                             {768.F, 0.F, 896.F, hf}}}}},
       {.id = 22,
        .name = "horizontal-stripes-3-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 64.F, hf},
                             {128.F, 0.F, 192.F, hf},
                             {256.F, 0.F, 3.20e+2F, hf},
                             {384.F, 0.F, 448.F, hf},
                             {512.F, 0.F, 576.F, hf},
                             {6.40e+2F, 0.F, 704.F, hf},
                             {768.F, 0.F, 832.F, hf},
                             {896.F, 0.F, 9.60e+2F, hf}}}}},
       {.id = 23,
        .name = "horizontal-stripes-3-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 64.F, hf},
                             {128.F, 0.F, 192.F, hf},
                             {256.F, 0.F, 3.20e+2F, hf},
                             {384.F, 0.F, 448.F, hf},
                             {512.F, 0.F, 576.F, hf},
                             {6.40e+2F, 0.F, 704.F, hf},
                             {768.F, 0.F, 832.F, hf},
                             {896.F, 0.F, 9.60e+2F, hf}}}}},
       {.id = 24,
        .name = "horizontal-stripes-4-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 32.F, hf},
                             {64.F, 0.F, 96.F, hf},
                             {128.F, 0.F, 1.60e+2F, hf},
                             {192.F, 0.F, 224.F, hf},
                             {256.F, 0.F, 288.F, hf},
                             {3.20e+2F, 0.F, 352.F, hf},
                             {384.F, 0.F, 416.F, hf},
                             {448.F, 0.F, 4.80e+2F, hf},
                             {512.F, 0.F, 544.F, hf},
                             {576.F, 0.F, 608.F, hf},
                             {6.40e+2F, 0.F, 672.F, hf},
                             {704.F, 0.F, 736.F, hf},
                             {768.F, 0.F, 8.00e+2F, hf},
                             {832.F, 0.F, 864.F, hf},
                             {896.F, 0.F, 928.F, hf},
                             {9.60e+2F, 0.F, 992.F, hf}}}}},
       {.id = 25,
        .name = "horizontal-stripes-4-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 32.F, hf},
                             {64.F, 0.F, 96.F, hf},
                             {128.F, 0.F, 1.60e+2F, hf},
                             {192.F, 0.F, 224.F, hf},
                             {256.F, 0.F, 288.F, hf},
                             {3.20e+2F, 0.F, 352.F, hf},
                             {384.F, 0.F, 416.F, hf},
                             {448.F, 0.F, 4.80e+2F, hf},
                             {512.F, 0.F, 544.F, hf},
                             {576.F, 0.F, 608.F, hf},
                             {6.40e+2F, 0.F, 672.F, hf},
                             {704.F, 0.F, 736.F, hf},
                             {768.F, 0.F, 8.00e+2F, hf},
                             {832.F, 0.F, 864.F, hf},
                             {896.F, 0.F, 928.F, hf},
                             {9.60e+2F, 0.F, 992.F, hf}}}}},
       {.id = 26,
        .name = "horizontal-stripes-5-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords =
                      {{0.F, 0.F, 16.F, hf},       {32.F, 0.F, 48.F, hf},
                       {64.F, 0.F, 80.F, hf},      {96.F, 0.F, 112.F, hf},
                       {128.F, 0.F, 144.F, hf},    {1.60e+2F, 0.F, 176.F, hf},
                       {192.F, 0.F, 208.F, hf},    {224.F, 0.F, 2.40e+2F, hf},
                       {256.F, 0.F, 272.F, hf},    {288.F, 0.F, 304.F, hf},
                       {3.20e+2F, 0.F, 336.F, hf}, {352.F, 0.F, 368.F, hf},
                       {384.F, 0.F, 4.00e+2F, hf}, {416.F, 0.F, 432.F, hf},
                       {448.F, 0.F, 464.F, hf},    {4.80e+2F, 0.F, 496.F, hf},
                       {512.F, 0.F, 528.F, hf},    {544.F, 0.F, 5.60e+2F, hf},
                       {576.F, 0.F, 592.F, hf},    {608.F, 0.F, 624.F, hf},
                       {6.40e+2F, 0.F, 656.F, hf}, {672.F, 0.F, 688.F, hf},
                       {704.F, 0.F, 7.20e+2F, hf}, {736.F, 0.F, 752.F, hf},
                       {768.F, 0.F, 784.F, hf},    {8.00e+2F, 0.F, 816.F, hf},
                       {832.F, 0.F, 848.F, hf},    {864.F, 0.F, 8.80e+2F, hf},
                       {896.F, 0.F, 912.F, hf},    {928.F, 0.F, 944.F, hf},
                       {9.60e+2F, 0.F, 976.F, hf}, {992.F, 0.F, 1008.F, hf}}}}},
       {.id = 27,
        .name = "horizontal-stripes-5-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords =
                      {{0.F, 0.F, 16.F, hf},       {32.F, 0.F, 48.F, hf},
                       {64.F, 0.F, 80.F, hf},      {96.F, 0.F, 112.F, hf},
                       {128.F, 0.F, 144.F, hf},    {1.60e+2F, 0.F, 176.F, hf},
                       {192.F, 0.F, 208.F, hf},    {224.F, 0.F, 2.40e+2F, hf},
                       {256.F, 0.F, 272.F, hf},    {288.F, 0.F, 304.F, hf},
                       {3.20e+2F, 0.F, 336.F, hf}, {352.F, 0.F, 368.F, hf},
                       {384.F, 0.F, 4.00e+2F, hf}, {416.F, 0.F, 432.F, hf},
                       {448.F, 0.F, 464.F, hf},    {4.80e+2F, 0.F, 496.F, hf},
                       {512.F, 0.F, 528.F, hf},    {544.F, 0.F, 5.60e+2F, hf},
                       {576.F, 0.F, 592.F, hf},    {608.F, 0.F, 624.F, hf},
                       {6.40e+2F, 0.F, 656.F, hf}, {672.F, 0.F, 688.F, hf},
                       {704.F, 0.F, 7.20e+2F, hf}, {736.F, 0.F, 752.F, hf},
                       {768.F, 0.F, 784.F, hf},    {8.00e+2F, 0.F, 816.F, hf},
                       {832.F, 0.F, 848.F, hf},    {864.F, 0.F, 8.80e+2F, hf},
                       {896.F, 0.F, 912.F, hf},    {928.F, 0.F, 944.F, hf},
                       {9.60e+2F, 0.F, 976.F, hf}, {992.F, 0.F, 1008.F, hf}}}}},
       {.id = 28,
        .name = "horizontal-stripes-6-normal",
        .draw =
            {{.color = {dark, dark, dark},
              .type =
                  GL_QUADS,
              .coords = {{0.F, 0.F, wf, hf}}},
             {.color = {bright, bright, bright},
              .type =
                  GL_QUADS,
              .coords =
                  {{0.F, 0.F, 8.0F, hf},       {16.F, 0.F, 24.F, hf},
                   {32.F, 0.F, 40.F, hf},      {48.F, 0.F, 56.F, hf},
                   {64.F, 0.F, 72.F, hf},      {80.F, 0.F, 88.F, hf},
                   {96.F, 0.F, 104.F, hf},     {112.F, 0.F, 1.20e+2F, hf},
                   {128.F, 0.F, 136.F, hf},    {144.F, 0.F, 152.F, hf},
                   {1.60e+2F, 0.F, 168.F, hf}, {176.F, 0.F, 184.F, hf},
                   {192.F, 0.F, 2.00e+2F, hf}, {208.F, 0.F, 216.F, hf},
                   {224.F, 0.F, 232.F, hf},    {2.40e+2F, 0.F, 248.F, hf},
                   {256.F, 0.F, 264.F, hf},    {272.F, 0.F, 2.80e+2F, hf},
                   {288.F, 0.F, 296.F, hf},    {304.F, 0.F, 312.F, hf},
                   {3.20e+2F, 0.F, 328.F, hf}, {336.F, 0.F, 344.F, hf},
                   {352.F, 0.F, 3.60e+2F, hf}, {368.F, 0.F, 376.F, hf},
                   {384.F, 0.F, 392.F, hf},    {4.00e+2F, 0.F, 408.F, hf},
                   {416.F, 0.F, 424.F, hf},    {432.F, 0.F, 4.40e+2F, hf},
                   {448.F, 0.F, 456.F, hf},    {464.F, 0.F, 472.F, hf},
                   {4.80e+2F, 0.F, 488.F, hf}, {496.F, 0.F, 504.F, hf},
                   {512.F, 0.F, 5.20e+2F, hf}, {528.F, 0.F, 536.F, hf},
                   {544.F, 0.F, 552.F, hf},    {5.60e+2F, 0.F, 568.F, hf},
                   {576.F, 0.F, 584.F, hf},    {592.F, 0.F, 6.00e+2F, hf},
                   {608.F, 0.F, 616.F, hf},    {624.F, 0.F, 632.F, hf},
                   {6.40e+2F, 0.F, 648.F, hf}, {656.F, 0.F, 664.F, hf},
                   {672.F, 0.F, 6.80e+2F, hf}, {688.F, 0.F, 696.F, hf},
                   {704.F, 0.F, 712.F, hf},    {7.20e+2F, 0.F, 728.F, hf},
                   {736.F, 0.F, 744.F, hf},    {752.F, 0.F, 7.60e+2F, hf},
                   {768.F, 0.F, 776.F, hf},    {784.F, 0.F, 792.F, hf},
                   {8.00e+2F, 0.F, 808.F, hf}, {816.F, 0.F, 824.F, hf},
                   {832.F, 0.F, 8.40e+2F, hf}, {848.F, 0.F, 856.F, hf},
                   {864.F, 0.F, 872.F, hf},    {8.80e+2F, 0.F, 888.F, hf},
                   {896.F, 0.F, 904.F, hf},    {912.F, 0.F, 9.20e+2F, hf},
                   {928.F, 0.F, 936.F, hf},    {944.F, 0.F, 952.F, hf},
                   {9.60e+2F, 0.F, 968.F, hf}, {976.F, 0.F, 984.F, hf},
                   {992.F, 0.F, 1.00e+3F, hf}, {1008.F, 0.F, 1016.F, hf}}}}},
       {.id = 29,
        .name = "horizontal-stripes-6-invert",
        .draw = {
            {.color = {bright, bright, bright},
             .type = GL_QUADS,
             .coords = {{0.F, 0.F, wf, hf}}},
            {.color = {dark, dark, dark},
             .type = GL_QUADS,
             .coords = {
                 {0.F, 0.F, 8.0F, hf},       {16.F, 0.F, 24.F, hf},
                 {32.F, 0.F, 40.F, hf},      {48.F, 0.F, 56.F, hf},
                 {64.F, 0.F, 72.F, hf},      {80.F, 0.F, 88.F, hf},
                 {96.F, 0.F, 104.F, hf},     {112.F, 0.F, 1.20e+2F, hf},
                 {128.F, 0.F, 136.F, hf},    {144.F, 0.F, 152.F, hf},
                 {1.60e+2F, 0.F, 168.F, hf}, {176.F, 0.F, 184.F, hf},
                 {192.F, 0.F, 2.00e+2F, hf}, {208.F, 0.F, 216.F, hf},
                 {224.F, 0.F, 232.F, hf},    {2.40e+2F, 0.F, 248.F, hf},
                 {256.F, 0.F, 264.F, hf},    {272.F, 0.F, 2.80e+2F, hf},
                 {288.F, 0.F, 296.F, hf},    {304.F, 0.F, 312.F, hf},
                 {3.20e+2F, 0.F, 328.F, hf}, {336.F, 0.F, 344.F, hf},
                 {352.F, 0.F, 3.60e+2F, hf}, {368.F, 0.F, 376.F, hf},
                 {384.F, 0.F, 392.F, hf},    {4.00e+2F, 0.F, 408.F, hf},
                 {416.F, 0.F, 424.F, hf},    {432.F, 0.F, 4.40e+2F, hf},
                 {448.F, 0.F, 456.F, hf},    {464.F, 0.F, 472.F, hf},
                 {4.80e+2F, 0.F, 488.F, hf}, {496.F, 0.F, 504.F, hf},
                 {512.F, 0.F, 5.20e+2F, hf}, {528.F, 0.F, 536.F, hf},
                 {544.F, 0.F, 552.F, hf},    {5.60e+2F, 0.F, 568.F, hf},
                 {576.F, 0.F, 584.F, hf},    {592.F, 0.F, 6.00e+2F, hf},
                 {608.F, 0.F, 616.F, hf},    {624.F, 0.F, 632.F, hf},
                 {6.40e+2F, 0.F, 648.F, hf}, {656.F, 0.F, 664.F, hf},
                 {672.F, 0.F, 6.80e+2F, hf}, {688.F, 0.F, 696.F, hf},
                 {704.F, 0.F, 712.F, hf},    {7.20e+2F, 0.F, 728.F, hf},
                 {736.F, 0.F, 744.F, hf},    {752.F, 0.F, 7.60e+2F, hf},
                 {768.F, 0.F, 776.F, hf},    {784.F, 0.F, 792.F, hf},
                 {8.00e+2F, 0.F, 808.F, hf}, {816.F, 0.F, 824.F, hf},
                 {832.F, 0.F, 8.40e+2F, hf}, {848.F, 0.F, 856.F, hf},
                 {864.F, 0.F, 872.F, hf},    {8.80e+2F, 0.F, 888.F, hf},
                 {896.F, 0.F, 904.F, hf},    {912.F, 0.F, 9.20e+2F, hf},
                 {928.F, 0.F, 936.F, hf},    {944.F, 0.F, 952.F, hf},
                 {9.60e+2F, 0.F, 968.F, hf}, {976.F, 0.F, 984.F, hf},
                 {992.F, 0.F, 1.00e+3F, hf}, {1008.F, 0.F, 1016.F, hf}}}}}};
  auto GLFW{glfw::init()};
  auto hints{glfw::WindowHints{.clientApi = glfw::ClientApi::OpenGl,
                               .contextVersionMajor = 2,
                               .contextVersionMinor = 0}};
  hints.apply();
  auto idStripeWidth{16};
  auto idBits{9};
  auto wId{idStripeWidth * idBits};
  auto wAll{w + wId};
  auto window{glfw::Window(wAll, h, "GLFWPP Grating")};
  glfw::makeContextCurrent(window);
  // an alternative to increase swap interval is to change screen update rate
  // `xrandr --output HDMI-A-0 --mode 1920x1080 --rate 24`
  glfw::swapInterval(swapInterval);
  auto frameId{0};
  while (!window.shouldClose()) {
    auto time{glfw::getTime()};
    glfw::pollEvents();
    if (frameId < drawFrames.size()) {
      frameId++;
    } else {
      frameId = 0;
    }
    glPushMatrix();
    // scale coordinates so that 0..w-1, 0..h-1 cover the screen
    glTranslatef(-1.0F, -1.0F, 0.F);
    glScalef(2.0F / wAll, 2.0F / h, 1.0F);
    drawFrames[frameId].execute();
    // green on black barcode for the id on the right
    glColor4f(0.F, 0.F, 0.F, 1.0F);
    glBegin(GL_QUADS);
    glVertex2f(w, 0);
    glVertex2f(wAll, 0);
    glVertex2f(wAll, h);
    glVertex2f(w, h);
    glEnd();
    glColor4f(0.F, 1.0F, 0.F, 1.0F);
    glBegin(GL_QUADS);
    static unsigned char id = 0;
    id++;
    for (decltype(0 + 8 + 1) i = 0; i < 8; i += 1) {
      if ((id & 1 << i)) {
        auto x0{w + i * idStripeWidth};
        auto x1{(w + (1 + i) * idStripeWidth) - (idStripeWidth / 2)};
        glVertex2f(x0, 0);
        glVertex2f(x1, 0);
        glVertex2f(x1, h);
        glVertex2f(x0, h);
      }
    }
    glEnd();
    glPopMatrix();
    window.swapBuffers();
    frameDelayEstimator.update();
  }
  return 0;
}
