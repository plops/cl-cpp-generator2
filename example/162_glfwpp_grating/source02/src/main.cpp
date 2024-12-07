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
                  .coords = {{0, 0, wf, hf}}}}},
       {.id = 1,
        .name = "all-dark",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0, 0, wf, hf}}}}},
       {.id = 2,
        .name = "vertical-stripes-0-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0, 0, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0, 0, 512, hf}}}}},
       {.id = 3,
        .name = "vertical-stripes-0-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0, 0, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0, 0, 512, hf}}}}},
       {.id = 4,
        .name = "vertical-stripes-1-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0, 0, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0, 0, 256, hf}, {512, 0, 768, hf}}}}},
       {.id = 5,
        .name = "vertical-stripes-1-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0, 0, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0, 0, 256, hf}, {512, 0, 768, hf}}}}},
       {.id = 6,
        .name = "vertical-stripes-2-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0, 0, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0, 0, 128, hf},
                             {256, 0, 384, hf},
                             {512, 0, 640, hf},
                             {768, 0, 896, hf}}}}},
       {.id = 7,
        .name = "vertical-stripes-2-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0, 0, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0, 0, 128, hf},
                             {256, 0, 384, hf},
                             {512, 0, 640, hf},
                             {768, 0, 896, hf}}}}},
       {.id = 8,
        .name = "vertical-stripes-3-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0, 0, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0, 0, 64, hf},
                             {128, 0, 192, hf},
                             {256, 0, 320, hf},
                             {384, 0, 448, hf},
                             {512, 0, 576, hf},
                             {640, 0, 704, hf},
                             {768, 0, 832, hf},
                             {896, 0, 960, hf}}}}},
       {.id = 9,
        .name = "vertical-stripes-3-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0, 0, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0, 0, 64, hf},
                             {128, 0, 192, hf},
                             {256, 0, 320, hf},
                             {384, 0, 448, hf},
                             {512, 0, 576, hf},
                             {640, 0, 704, hf},
                             {768, 0, 832, hf},
                             {896, 0, 960, hf}}}}},
       {.id = 10,
        .name = "vertical-stripes-4-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0, 0, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0, 0, 32, hf},
                             {64, 0, 96, hf},
                             {128, 0, 160, hf},
                             {192, 0, 224, hf},
                             {256, 0, 288, hf},
                             {320, 0, 352, hf},
                             {384, 0, 416, hf},
                             {448, 0, 480, hf},
                             {512, 0, 544, hf},
                             {576, 0, 608, hf},
                             {640, 0, 672, hf},
                             {704, 0, 736, hf},
                             {768, 0, 800, hf},
                             {832, 0, 864, hf},
                             {896, 0, 928, hf},
                             {960, 0, 992, hf}}}}},
       {.id = 11,
        .name = "vertical-stripes-4-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0, 0, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0, 0, 32, hf},
                             {64, 0, 96, hf},
                             {128, 0, 160, hf},
                             {192, 0, 224, hf},
                             {256, 0, 288, hf},
                             {320, 0, 352, hf},
                             {384, 0, 416, hf},
                             {448, 0, 480, hf},
                             {512, 0, 544, hf},
                             {576, 0, 608, hf},
                             {640, 0, 672, hf},
                             {704, 0, 736, hf},
                             {768, 0, 800, hf},
                             {832, 0, 864, hf},
                             {896, 0, 928, hf},
                             {960, 0, 992, hf}}}}},
       {.id = 12,
        .name = "vertical-stripes-5-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0, 0, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0, 0, 16, hf},    {32, 0, 48, hf},
                             {64, 0, 80, hf},   {96, 0, 112, hf},
                             {128, 0, 144, hf}, {160, 0, 176, hf},
                             {192, 0, 208, hf}, {224, 0, 240, hf},
                             {256, 0, 272, hf}, {288, 0, 304, hf},
                             {320, 0, 336, hf}, {352, 0, 368, hf},
                             {384, 0, 400, hf}, {416, 0, 432, hf},
                             {448, 0, 464, hf}, {480, 0, 496, hf},
                             {512, 0, 528, hf}, {544, 0, 560, hf},
                             {576, 0, 592, hf}, {608, 0, 624, hf},
                             {640, 0, 656, hf}, {672, 0, 688, hf},
                             {704, 0, 720, hf}, {736, 0, 752, hf},
                             {768, 0, 784, hf}, {800, 0, 816, hf},
                             {832, 0, 848, hf}, {864, 0, 880, hf},
                             {896, 0, 912, hf}, {928, 0, 944, hf},
                             {960, 0, 976, hf}, {992, 0, 1008, hf}}}}},
       {.id = 13,
        .name = "vertical-stripes-5-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0, 0, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0, 0, 16, hf},    {32, 0, 48, hf},
                             {64, 0, 80, hf},   {96, 0, 112, hf},
                             {128, 0, 144, hf}, {160, 0, 176, hf},
                             {192, 0, 208, hf}, {224, 0, 240, hf},
                             {256, 0, 272, hf}, {288, 0, 304, hf},
                             {320, 0, 336, hf}, {352, 0, 368, hf},
                             {384, 0, 400, hf}, {416, 0, 432, hf},
                             {448, 0, 464, hf}, {480, 0, 496, hf},
                             {512, 0, 528, hf}, {544, 0, 560, hf},
                             {576, 0, 592, hf}, {608, 0, 624, hf},
                             {640, 0, 656, hf}, {672, 0, 688, hf},
                             {704, 0, 720, hf}, {736, 0, 752, hf},
                             {768, 0, 784, hf}, {800, 0, 816, hf},
                             {832, 0, 848, hf}, {864, 0, 880, hf},
                             {896, 0, 912, hf}, {928, 0, 944, hf},
                             {960, 0, 976, hf}, {992, 0, 1008, hf}}}}},
       {.id = 14,
        .name = "vertical-stripes-6-normal",
        .draw =
            {{.color = {dark, dark, dark},
              .type =
                  GL_QUADS,
              .coords = {{0, 0, wf, hf}}},
             {.color = {dark, dark, dark},
              .type =
                  GL_QUADS,
              .coords =
                  {{0, 0, 8, hf},      {16, 0, 24, hf},   {32, 0, 40, hf},
                   {48, 0, 56, hf},    {64, 0, 72, hf},   {80, 0, 88, hf},
                   {96, 0, 104, hf},   {112, 0, 120, hf}, {128, 0, 136, hf},
                   {144, 0, 152, hf},  {160, 0, 168, hf}, {176, 0, 184, hf},
                   {192, 0, 200, hf},  {208, 0, 216, hf}, {224, 0, 232, hf},
                   {240, 0, 248, hf},  {256, 0, 264, hf}, {272, 0, 280, hf},
                   {288, 0, 296, hf},  {304, 0, 312, hf}, {320, 0, 328, hf},
                   {336, 0, 344, hf},  {352, 0, 360, hf}, {368, 0, 376, hf},
                   {384, 0, 392, hf},  {400, 0, 408, hf}, {416, 0, 424, hf},
                   {432, 0, 440, hf},  {448, 0, 456, hf}, {464, 0, 472, hf},
                   {480, 0, 488, hf},  {496, 0, 504, hf}, {512, 0, 520, hf},
                   {528, 0, 536, hf},  {544, 0, 552, hf}, {560, 0, 568, hf},
                   {576, 0, 584, hf},  {592, 0, 600, hf}, {608, 0, 616, hf},
                   {624, 0, 632, hf},  {640, 0, 648, hf}, {656, 0, 664, hf},
                   {672, 0, 680, hf},  {688, 0, 696, hf}, {704, 0, 712, hf},
                   {720, 0, 728, hf},  {736, 0, 744, hf}, {752, 0, 760, hf},
                   {768, 0, 776, hf},  {784, 0, 792, hf}, {800, 0, 808, hf},
                   {816, 0, 824, hf},  {832, 0, 840, hf}, {848, 0, 856, hf},
                   {864, 0, 872, hf},  {880, 0, 888, hf}, {896, 0, 904, hf},
                   {912, 0, 920, hf},  {928, 0, 936, hf}, {944, 0, 952, hf},
                   {960, 0, 968, hf},  {976, 0, 984, hf}, {992, 0, 1000, hf},
                   {1008, 0, 1016, hf}}}}},
       {.id = 15,
        .name = "vertical-stripes-6-invert",
        .draw =
            {{.color = {bright, bright, bright},
              .type =
                  GL_QUADS,
              .coords = {{0, 0, wf, hf}}},
             {.color = {bright, bright, bright},
              .type =
                  GL_QUADS,
              .coords =
                  {{0, 0, 8, hf},      {16, 0, 24, hf},   {32, 0, 40, hf},
                   {48, 0, 56, hf},    {64, 0, 72, hf},   {80, 0, 88, hf},
                   {96, 0, 104, hf},   {112, 0, 120, hf}, {128, 0, 136, hf},
                   {144, 0, 152, hf},  {160, 0, 168, hf}, {176, 0, 184, hf},
                   {192, 0, 200, hf},  {208, 0, 216, hf}, {224, 0, 232, hf},
                   {240, 0, 248, hf},  {256, 0, 264, hf}, {272, 0, 280, hf},
                   {288, 0, 296, hf},  {304, 0, 312, hf}, {320, 0, 328, hf},
                   {336, 0, 344, hf},  {352, 0, 360, hf}, {368, 0, 376, hf},
                   {384, 0, 392, hf},  {400, 0, 408, hf}, {416, 0, 424, hf},
                   {432, 0, 440, hf},  {448, 0, 456, hf}, {464, 0, 472, hf},
                   {480, 0, 488, hf},  {496, 0, 504, hf}, {512, 0, 520, hf},
                   {528, 0, 536, hf},  {544, 0, 552, hf}, {560, 0, 568, hf},
                   {576, 0, 584, hf},  {592, 0, 600, hf}, {608, 0, 616, hf},
                   {624, 0, 632, hf},  {640, 0, 648, hf}, {656, 0, 664, hf},
                   {672, 0, 680, hf},  {688, 0, 696, hf}, {704, 0, 712, hf},
                   {720, 0, 728, hf},  {736, 0, 744, hf}, {752, 0, 760, hf},
                   {768, 0, 776, hf},  {784, 0, 792, hf}, {800, 0, 808, hf},
                   {816, 0, 824, hf},  {832, 0, 840, hf}, {848, 0, 856, hf},
                   {864, 0, 872, hf},  {880, 0, 888, hf}, {896, 0, 904, hf},
                   {912, 0, 920, hf},  {928, 0, 936, hf}, {944, 0, 952, hf},
                   {960, 0, 968, hf},  {976, 0, 984, hf}, {992, 0, 1000, hf},
                   {1008, 0, 1016, hf}}}}},
       {.id = 16,
        .name = "horizontal-stripes-0-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0, 0, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0, 0, 512, hf}}}}},
       {.id = 17,
        .name = "horizontal-stripes-0-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0, 0, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0, 0, 512, hf}}}}},
       {.id = 18,
        .name = "horizontal-stripes-1-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0, 0, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0, 0, 256, hf}, {512, 0, 768, hf}}}}},
       {.id = 19,
        .name = "horizontal-stripes-1-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0, 0, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0, 0, 256, hf}, {512, 0, 768, hf}}}}},
       {.id = 20,
        .name = "horizontal-stripes-2-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0, 0, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0, 0, 128, hf},
                             {256, 0, 384, hf},
                             {512, 0, 640, hf},
                             {768, 0, 896, hf}}}}},
       {.id = 21,
        .name = "horizontal-stripes-2-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0, 0, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0, 0, 128, hf},
                             {256, 0, 384, hf},
                             {512, 0, 640, hf},
                             {768, 0, 896, hf}}}}},
       {.id = 22,
        .name = "horizontal-stripes-3-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0, 0, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0, 0, 64, hf},
                             {128, 0, 192, hf},
                             {256, 0, 320, hf},
                             {384, 0, 448, hf},
                             {512, 0, 576, hf},
                             {640, 0, 704, hf},
                             {768, 0, 832, hf},
                             {896, 0, 960, hf}}}}},
       {.id = 23,
        .name = "horizontal-stripes-3-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0, 0, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0, 0, 64, hf},
                             {128, 0, 192, hf},
                             {256, 0, 320, hf},
                             {384, 0, 448, hf},
                             {512, 0, 576, hf},
                             {640, 0, 704, hf},
                             {768, 0, 832, hf},
                             {896, 0, 960, hf}}}}},
       {.id = 24,
        .name = "horizontal-stripes-4-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0, 0, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0, 0, 32, hf},
                             {64, 0, 96, hf},
                             {128, 0, 160, hf},
                             {192, 0, 224, hf},
                             {256, 0, 288, hf},
                             {320, 0, 352, hf},
                             {384, 0, 416, hf},
                             {448, 0, 480, hf},
                             {512, 0, 544, hf},
                             {576, 0, 608, hf},
                             {640, 0, 672, hf},
                             {704, 0, 736, hf},
                             {768, 0, 800, hf},
                             {832, 0, 864, hf},
                             {896, 0, 928, hf},
                             {960, 0, 992, hf}}}}},
       {.id = 25,
        .name = "horizontal-stripes-4-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0, 0, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0, 0, 32, hf},
                             {64, 0, 96, hf},
                             {128, 0, 160, hf},
                             {192, 0, 224, hf},
                             {256, 0, 288, hf},
                             {320, 0, 352, hf},
                             {384, 0, 416, hf},
                             {448, 0, 480, hf},
                             {512, 0, 544, hf},
                             {576, 0, 608, hf},
                             {640, 0, 672, hf},
                             {704, 0, 736, hf},
                             {768, 0, 800, hf},
                             {832, 0, 864, hf},
                             {896, 0, 928, hf},
                             {960, 0, 992, hf}}}}},
       {.id = 26,
        .name = "horizontal-stripes-5-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0, 0, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords =
                      {{0, 0, 16, hf},    {32, 0, 48, hf},   {64, 0, 80, hf},
                       {96, 0, 112, hf},  {128, 0, 144, hf}, {160, 0, 176, hf},
                       {192, 0, 208, hf}, {224, 0, 240, hf}, {256, 0, 272, hf},
                       {288, 0, 304, hf}, {320, 0, 336, hf}, {352, 0, 368, hf},
                       {384, 0, 400, hf}, {416, 0, 432, hf}, {448, 0, 464, hf},
                       {480, 0, 496, hf}, {512, 0, 528, hf}, {544, 0, 560, hf},
                       {576, 0, 592, hf}, {608, 0, 624, hf}, {640, 0, 656, hf},
                       {672, 0, 688, hf}, {704, 0, 720, hf}, {736, 0, 752, hf},
                       {768, 0, 784, hf}, {800, 0, 816, hf}, {832, 0, 848, hf},
                       {864, 0, 880, hf}, {896, 0, 912, hf}, {928, 0, 944, hf},
                       {960, 0, 976, hf}, {992, 0, 1008, hf}}}}},
       {.id = 27,
        .name = "horizontal-stripes-5-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0, 0, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords =
                      {{0, 0, 16, hf},    {32, 0, 48, hf},   {64, 0, 80, hf},
                       {96, 0, 112, hf},  {128, 0, 144, hf}, {160, 0, 176, hf},
                       {192, 0, 208, hf}, {224, 0, 240, hf}, {256, 0, 272, hf},
                       {288, 0, 304, hf}, {320, 0, 336, hf}, {352, 0, 368, hf},
                       {384, 0, 400, hf}, {416, 0, 432, hf}, {448, 0, 464, hf},
                       {480, 0, 496, hf}, {512, 0, 528, hf}, {544, 0, 560, hf},
                       {576, 0, 592, hf}, {608, 0, 624, hf}, {640, 0, 656, hf},
                       {672, 0, 688, hf}, {704, 0, 720, hf}, {736, 0, 752, hf},
                       {768, 0, 784, hf}, {800, 0, 816, hf}, {832, 0, 848, hf},
                       {864, 0, 880, hf}, {896, 0, 912, hf}, {928, 0, 944, hf},
                       {960, 0, 976, hf}, {992, 0, 1008, hf}}}}},
       {.id = 28,
        .name = "horizontal-stripes-6-normal",
        .draw =
            {{.color = {dark, dark, dark},
              .type =
                  GL_QUADS,
              .coords = {{0, 0, wf, hf}}},
             {.color = {dark, dark, dark},
              .type =
                  GL_QUADS,
              .coords =
                  {{0, 0, 8, hf},      {16, 0, 24, hf},   {32, 0, 40, hf},
                   {48, 0, 56, hf},    {64, 0, 72, hf},   {80, 0, 88, hf},
                   {96, 0, 104, hf},   {112, 0, 120, hf}, {128, 0, 136, hf},
                   {144, 0, 152, hf},  {160, 0, 168, hf}, {176, 0, 184, hf},
                   {192, 0, 200, hf},  {208, 0, 216, hf}, {224, 0, 232, hf},
                   {240, 0, 248, hf},  {256, 0, 264, hf}, {272, 0, 280, hf},
                   {288, 0, 296, hf},  {304, 0, 312, hf}, {320, 0, 328, hf},
                   {336, 0, 344, hf},  {352, 0, 360, hf}, {368, 0, 376, hf},
                   {384, 0, 392, hf},  {400, 0, 408, hf}, {416, 0, 424, hf},
                   {432, 0, 440, hf},  {448, 0, 456, hf}, {464, 0, 472, hf},
                   {480, 0, 488, hf},  {496, 0, 504, hf}, {512, 0, 520, hf},
                   {528, 0, 536, hf},  {544, 0, 552, hf}, {560, 0, 568, hf},
                   {576, 0, 584, hf},  {592, 0, 600, hf}, {608, 0, 616, hf},
                   {624, 0, 632, hf},  {640, 0, 648, hf}, {656, 0, 664, hf},
                   {672, 0, 680, hf},  {688, 0, 696, hf}, {704, 0, 712, hf},
                   {720, 0, 728, hf},  {736, 0, 744, hf}, {752, 0, 760, hf},
                   {768, 0, 776, hf},  {784, 0, 792, hf}, {800, 0, 808, hf},
                   {816, 0, 824, hf},  {832, 0, 840, hf}, {848, 0, 856, hf},
                   {864, 0, 872, hf},  {880, 0, 888, hf}, {896, 0, 904, hf},
                   {912, 0, 920, hf},  {928, 0, 936, hf}, {944, 0, 952, hf},
                   {960, 0, 968, hf},  {976, 0, 984, hf}, {992, 0, 1000, hf},
                   {1008, 0, 1016, hf}}}}},
       {.id = 29,
        .name = "horizontal-stripes-6-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0, 0, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {
                      {0, 0, 8, hf},      {16, 0, 24, hf},   {32, 0, 40, hf},
                      {48, 0, 56, hf},    {64, 0, 72, hf},   {80, 0, 88, hf},
                      {96, 0, 104, hf},   {112, 0, 120, hf}, {128, 0, 136, hf},
                      {144, 0, 152, hf},  {160, 0, 168, hf}, {176, 0, 184, hf},
                      {192, 0, 200, hf},  {208, 0, 216, hf}, {224, 0, 232, hf},
                      {240, 0, 248, hf},  {256, 0, 264, hf}, {272, 0, 280, hf},
                      {288, 0, 296, hf},  {304, 0, 312, hf}, {320, 0, 328, hf},
                      {336, 0, 344, hf},  {352, 0, 360, hf}, {368, 0, 376, hf},
                      {384, 0, 392, hf},  {400, 0, 408, hf}, {416, 0, 424, hf},
                      {432, 0, 440, hf},  {448, 0, 456, hf}, {464, 0, 472, hf},
                      {480, 0, 488, hf},  {496, 0, 504, hf}, {512, 0, 520, hf},
                      {528, 0, 536, hf},  {544, 0, 552, hf}, {560, 0, 568, hf},
                      {576, 0, 584, hf},  {592, 0, 600, hf}, {608, 0, 616, hf},
                      {624, 0, 632, hf},  {640, 0, 648, hf}, {656, 0, 664, hf},
                      {672, 0, 680, hf},  {688, 0, 696, hf}, {704, 0, 712, hf},
                      {720, 0, 728, hf},  {736, 0, 744, hf}, {752, 0, 760, hf},
                      {768, 0, 776, hf},  {784, 0, 792, hf}, {800, 0, 808, hf},
                      {816, 0, 824, hf},  {832, 0, 840, hf}, {848, 0, 856, hf},
                      {864, 0, 872, hf},  {880, 0, 888, hf}, {896, 0, 904, hf},
                      {912, 0, 920, hf},  {928, 0, 936, hf}, {944, 0, 952, hf},
                      {960, 0, 968, hf},  {976, 0, 984, hf}, {992, 0, 1000, hf},
                      {1008, 0, 1016, hf}}}}}};
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
    auto draws{drawFrames[frameId].draw};
    for (auto &&[color, type, coords] : draws) {
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
