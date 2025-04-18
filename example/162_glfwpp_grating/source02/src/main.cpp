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

void drawBarcode(int code, int bits, int idStripeWidth, float x0, float x1,
                 float y0, float y1) {
  // green on black barcode for the id on the right
  glColor4f(0.F, 0.F, 0.F, 1.0F);
  glBegin(GL_QUADS);
  glVertex2f(x0, y0);
  glVertex2f(x1, y0);
  glVertex2f(x1, y1);
  glVertex2f(x0, y1);
  glEnd();
  glBegin(GL_QUADS);
  for (decltype(0 + bits + 1) i = 0; i < bits; i += 1) {
    // Hi bits are green, low bits are red
    if ((code & (1 << i))) {
      glColor4f(0.F, 1.0F, 0.F, 1.0F);
    } else {
      glColor4f(1.0F, 0.F, 0.F, 1.0F);
    }
    auto xx0{x0 + i * idStripeWidth};
    auto xx1{(x0 + (1 + i) * idStripeWidth) - (idStripeWidth / 2)};
    glVertex2f(xx0, y0);
    glVertex2f(xx1, y0);
    glVertex2f(xx1, y1);
    glVertex2f(xx0, y1);
  }
  glEnd();
}

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
  auto swapInterval{int(1)};
  auto numberFramesForStatistics{int(211)};
  auto darkLevel{int(0)};
  auto brightLevel{int(255)};
  auto helpOption{op.add<popl::Switch>("h", "help", "produce help message")};
  auto verboseOption{
      op.add<popl::Switch>("v", "verbose", "produce verbose output")};
  auto swapIntervalOption{op.add<popl::Value<int>>(
      "s", "swapInterval", "parameter", 1, &swapInterval)};
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
  auto w{1024};
  auto h{1024};
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
                  .coords = {{0.F, 0.F, 1024.F, hf}}}}},
       {.id = 3,
        .name = "vertical-stripes-0-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 1024.F, hf}}}}},
       {.id = 4,
        .name = "vertical-stripes-1-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 512.F, hf},
                             {1024.F, 0.F, 1536.F, hf}}}}},
       {.id = 5,
        .name = "vertical-stripes-1-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 512.F, hf},
                             {1024.F, 0.F, 1536.F, hf}}}}},
       {.id = 6,
        .name = "vertical-stripes-2-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 256.F, hf},
                             {512.F, 0.F, 768.F, hf},
                             {1024.F, 0.F, 1.280e+3F, hf},
                             {1536.F, 0.F, 1792.F, hf}}}}},
       {.id = 7,
        .name = "vertical-stripes-2-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 256.F, hf},
                             {512.F, 0.F, 768.F, hf},
                             {1024.F, 0.F, 1.280e+3F, hf},
                             {1536.F, 0.F, 1792.F, hf}}}}},
       {.id = 8,
        .name = "vertical-stripes-3-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 128.F, hf},
                             {256.F, 0.F, 384.F, hf},
                             {512.F, 0.F, 6.40e+2F, hf},
                             {768.F, 0.F, 896.F, hf},
                             {1024.F, 0.F, 1152.F, hf},
                             {1.280e+3F, 0.F, 1408.F, hf},
                             {1536.F, 0.F, 1664.F, hf},
                             {1792.F, 0.F, 1.920e+3F, hf}}}}},
       {.id = 9,
        .name = "vertical-stripes-3-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 128.F, hf},
                             {256.F, 0.F, 384.F, hf},
                             {512.F, 0.F, 6.40e+2F, hf},
                             {768.F, 0.F, 896.F, hf},
                             {1024.F, 0.F, 1152.F, hf},
                             {1.280e+3F, 0.F, 1408.F, hf},
                             {1536.F, 0.F, 1664.F, hf},
                             {1792.F, 0.F, 1.920e+3F, hf}}}}},
       {.id = 10,
        .name = "vertical-stripes-4-normal",
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
                             {896.F, 0.F, 9.60e+2F, hf},
                             {1024.F, 0.F, 1088.F, hf},
                             {1152.F, 0.F, 1216.F, hf},
                             {1.280e+3F, 0.F, 1344.F, hf},
                             {1408.F, 0.F, 1472.F, hf},
                             {1536.F, 0.F, 1.60e+3F, hf},
                             {1664.F, 0.F, 1728.F, hf},
                             {1792.F, 0.F, 1856.F, hf},
                             {1.920e+3F, 0.F, 1984.F, hf}}}}},
       {.id = 11,
        .name = "vertical-stripes-4-invert",
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
                             {896.F, 0.F, 9.60e+2F, hf},
                             {1024.F, 0.F, 1088.F, hf},
                             {1152.F, 0.F, 1216.F, hf},
                             {1.280e+3F, 0.F, 1344.F, hf},
                             {1408.F, 0.F, 1472.F, hf},
                             {1536.F, 0.F, 1.60e+3F, hf},
                             {1664.F, 0.F, 1728.F, hf},
                             {1792.F, 0.F, 1856.F, hf},
                             {1.920e+3F, 0.F, 1984.F, hf}}}}},
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
                  {{0.F, 0.F, 32.F, hf},         {64.F, 0.F, 96.F, hf},
                   {128.F, 0.F, 1.60e+2F, hf},   {192.F, 0.F, 224.F, hf},
                   {256.F, 0.F, 288.F, hf},      {3.20e+2F, 0.F, 352.F, hf},
                   {384.F, 0.F, 416.F, hf},      {448.F, 0.F, 4.80e+2F, hf},
                   {512.F, 0.F, 544.F, hf},      {576.F, 0.F, 608.F, hf},
                   {6.40e+2F, 0.F, 672.F, hf},   {704.F, 0.F, 736.F, hf},
                   {768.F, 0.F, 8.00e+2F, hf},   {832.F, 0.F, 864.F, hf},
                   {896.F, 0.F, 928.F, hf},      {9.60e+2F, 0.F, 992.F, hf},
                   {1024.F, 0.F, 1056.F, hf},    {1088.F, 0.F, 1.120e+3F, hf},
                   {1152.F, 0.F, 1184.F, hf},    {1216.F, 0.F, 1248.F, hf},
                   {1.280e+3F, 0.F, 1312.F, hf}, {1344.F, 0.F, 1376.F, hf},
                   {1408.F, 0.F, 1.440e+3F, hf}, {1472.F, 0.F, 1504.F, hf},
                   {1536.F, 0.F, 1568.F, hf},    {1.60e+3F, 0.F, 1632.F, hf},
                   {1664.F, 0.F, 1696.F, hf},    {1728.F, 0.F, 1.760e+3F, hf},
                   {1792.F, 0.F, 1824.F, hf},    {1856.F, 0.F, 1888.F, hf},
                   {1.920e+3F, 0.F, 1952.F, hf}, {1984.F, 0.F, 2016.F, hf}}}}},
       {.id = 13,
        .name = "vertical-stripes-5-invert",
        .draw =
            {{.color = {bright, bright, bright},
              .type =
                  GL_QUADS,
              .coords = {{0.F, 0.F, wf, hf}}},
             {.color = {dark, dark, dark},
              .type =
                  GL_QUADS,
              .coords =
                  {{0.F, 0.F, 32.F, hf},         {64.F, 0.F, 96.F, hf},
                   {128.F, 0.F, 1.60e+2F, hf},   {192.F, 0.F, 224.F, hf},
                   {256.F, 0.F, 288.F, hf},      {3.20e+2F, 0.F, 352.F, hf},
                   {384.F, 0.F, 416.F, hf},      {448.F, 0.F, 4.80e+2F, hf},
                   {512.F, 0.F, 544.F, hf},      {576.F, 0.F, 608.F, hf},
                   {6.40e+2F, 0.F, 672.F, hf},   {704.F, 0.F, 736.F, hf},
                   {768.F, 0.F, 8.00e+2F, hf},   {832.F, 0.F, 864.F, hf},
                   {896.F, 0.F, 928.F, hf},      {9.60e+2F, 0.F, 992.F, hf},
                   {1024.F, 0.F, 1056.F, hf},    {1088.F, 0.F, 1.120e+3F, hf},
                   {1152.F, 0.F, 1184.F, hf},    {1216.F, 0.F, 1248.F, hf},
                   {1.280e+3F, 0.F, 1312.F, hf}, {1344.F, 0.F, 1376.F, hf},
                   {1408.F, 0.F, 1.440e+3F, hf}, {1472.F, 0.F, 1504.F, hf},
                   {1536.F, 0.F, 1568.F, hf},    {1.60e+3F, 0.F, 1632.F, hf},
                   {1664.F, 0.F, 1696.F, hf},    {1728.F, 0.F, 1.760e+3F, hf},
                   {1792.F, 0.F, 1824.F, hf},    {1856.F, 0.F, 1888.F, hf},
                   {1.920e+3F, 0.F, 1952.F, hf}, {1984.F, 0.F, 2016.F, hf}}}}},
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
                  {{0.F, 0.F, 16.F, hf},         {32.F, 0.F, 48.F, hf},
                   {64.F, 0.F, 80.F, hf},        {96.F, 0.F, 112.F, hf},
                   {128.F, 0.F, 144.F, hf},      {1.60e+2F, 0.F, 176.F, hf},
                   {192.F, 0.F, 208.F, hf},      {224.F, 0.F, 2.40e+2F, hf},
                   {256.F, 0.F, 272.F, hf},      {288.F, 0.F, 304.F, hf},
                   {3.20e+2F, 0.F, 336.F, hf},   {352.F, 0.F, 368.F, hf},
                   {384.F, 0.F, 4.00e+2F, hf},   {416.F, 0.F, 432.F, hf},
                   {448.F, 0.F, 464.F, hf},      {4.80e+2F, 0.F, 496.F, hf},
                   {512.F, 0.F, 528.F, hf},      {544.F, 0.F, 5.60e+2F, hf},
                   {576.F, 0.F, 592.F, hf},      {608.F, 0.F, 624.F, hf},
                   {6.40e+2F, 0.F, 656.F, hf},   {672.F, 0.F, 688.F, hf},
                   {704.F, 0.F, 7.20e+2F, hf},   {736.F, 0.F, 752.F, hf},
                   {768.F, 0.F, 784.F, hf},      {8.00e+2F, 0.F, 816.F, hf},
                   {832.F, 0.F, 848.F, hf},      {864.F, 0.F, 8.80e+2F, hf},
                   {896.F, 0.F, 912.F, hf},      {928.F, 0.F, 944.F, hf},
                   {9.60e+2F, 0.F, 976.F, hf},   {992.F, 0.F, 1008.F, hf},
                   {1024.F, 0.F, 1.040e+3F, hf}, {1056.F, 0.F, 1072.F, hf},
                   {1088.F, 0.F, 1104.F, hf},    {1.120e+3F, 0.F, 1136.F, hf},
                   {1152.F, 0.F, 1168.F, hf},    {1184.F, 0.F, 1.20e+3F, hf},
                   {1216.F, 0.F, 1232.F, hf},    {1248.F, 0.F, 1264.F, hf},
                   {1.280e+3F, 0.F, 1296.F, hf}, {1312.F, 0.F, 1328.F, hf},
                   {1344.F, 0.F, 1.360e+3F, hf}, {1376.F, 0.F, 1392.F, hf},
                   {1408.F, 0.F, 1424.F, hf},    {1.440e+3F, 0.F, 1456.F, hf},
                   {1472.F, 0.F, 1488.F, hf},    {1504.F, 0.F, 1.520e+3F, hf},
                   {1536.F, 0.F, 1552.F, hf},    {1568.F, 0.F, 1584.F, hf},
                   {1.60e+3F, 0.F, 1616.F, hf},  {1632.F, 0.F, 1648.F, hf},
                   {1664.F, 0.F, 1.680e+3F, hf}, {1696.F, 0.F, 1712.F, hf},
                   {1728.F, 0.F, 1744.F, hf},    {1.760e+3F, 0.F, 1776.F, hf},
                   {1792.F, 0.F, 1808.F, hf},    {1824.F, 0.F, 1.840e+3F, hf},
                   {1856.F, 0.F, 1872.F, hf},    {1888.F, 0.F, 1904.F, hf},
                   {1.920e+3F, 0.F, 1936.F, hf}, {1952.F, 0.F, 1968.F, hf},
                   {1984.F, 0.F, 2.00e+3F, hf},  {2016.F, 0.F, 2032.F, hf}}}}},
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
                  {{0.F, 0.F, 16.F, hf},         {32.F, 0.F, 48.F, hf},
                   {64.F, 0.F, 80.F, hf},        {96.F, 0.F, 112.F, hf},
                   {128.F, 0.F, 144.F, hf},      {1.60e+2F, 0.F, 176.F, hf},
                   {192.F, 0.F, 208.F, hf},      {224.F, 0.F, 2.40e+2F, hf},
                   {256.F, 0.F, 272.F, hf},      {288.F, 0.F, 304.F, hf},
                   {3.20e+2F, 0.F, 336.F, hf},   {352.F, 0.F, 368.F, hf},
                   {384.F, 0.F, 4.00e+2F, hf},   {416.F, 0.F, 432.F, hf},
                   {448.F, 0.F, 464.F, hf},      {4.80e+2F, 0.F, 496.F, hf},
                   {512.F, 0.F, 528.F, hf},      {544.F, 0.F, 5.60e+2F, hf},
                   {576.F, 0.F, 592.F, hf},      {608.F, 0.F, 624.F, hf},
                   {6.40e+2F, 0.F, 656.F, hf},   {672.F, 0.F, 688.F, hf},
                   {704.F, 0.F, 7.20e+2F, hf},   {736.F, 0.F, 752.F, hf},
                   {768.F, 0.F, 784.F, hf},      {8.00e+2F, 0.F, 816.F, hf},
                   {832.F, 0.F, 848.F, hf},      {864.F, 0.F, 8.80e+2F, hf},
                   {896.F, 0.F, 912.F, hf},      {928.F, 0.F, 944.F, hf},
                   {9.60e+2F, 0.F, 976.F, hf},   {992.F, 0.F, 1008.F, hf},
                   {1024.F, 0.F, 1.040e+3F, hf}, {1056.F, 0.F, 1072.F, hf},
                   {1088.F, 0.F, 1104.F, hf},    {1.120e+3F, 0.F, 1136.F, hf},
                   {1152.F, 0.F, 1168.F, hf},    {1184.F, 0.F, 1.20e+3F, hf},
                   {1216.F, 0.F, 1232.F, hf},    {1248.F, 0.F, 1264.F, hf},
                   {1.280e+3F, 0.F, 1296.F, hf}, {1312.F, 0.F, 1328.F, hf},
                   {1344.F, 0.F, 1.360e+3F, hf}, {1376.F, 0.F, 1392.F, hf},
                   {1408.F, 0.F, 1424.F, hf},    {1.440e+3F, 0.F, 1456.F, hf},
                   {1472.F, 0.F, 1488.F, hf},    {1504.F, 0.F, 1.520e+3F, hf},
                   {1536.F, 0.F, 1552.F, hf},    {1568.F, 0.F, 1584.F, hf},
                   {1.60e+3F, 0.F, 1616.F, hf},  {1632.F, 0.F, 1648.F, hf},
                   {1664.F, 0.F, 1.680e+3F, hf}, {1696.F, 0.F, 1712.F, hf},
                   {1728.F, 0.F, 1744.F, hf},    {1.760e+3F, 0.F, 1776.F, hf},
                   {1792.F, 0.F, 1808.F, hf},    {1824.F, 0.F, 1.840e+3F, hf},
                   {1856.F, 0.F, 1872.F, hf},    {1888.F, 0.F, 1904.F, hf},
                   {1.920e+3F, 0.F, 1936.F, hf}, {1952.F, 0.F, 1968.F, hf},
                   {1984.F, 0.F, 2.00e+3F, hf},  {2016.F, 0.F, 2032.F, hf}}}}},
       {.id = 16,
        .name = "vertical-stripes-7-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 8.0F, hf},
                             {16.F, 0.F, 24.F, hf},
                             {32.F, 0.F, 40.F, hf},
                             {48.F, 0.F, 56.F, hf},
                             {64.F, 0.F, 72.F, hf},
                             {80.F, 0.F, 88.F, hf},
                             {96.F, 0.F, 104.F, hf},
                             {112.F, 0.F, 1.20e+2F, hf},
                             {128.F, 0.F, 136.F, hf},
                             {144.F, 0.F, 152.F, hf},
                             {1.60e+2F, 0.F, 168.F, hf},
                             {176.F, 0.F, 184.F, hf},
                             {192.F, 0.F, 2.00e+2F, hf},
                             {208.F, 0.F, 216.F, hf},
                             {224.F, 0.F, 232.F, hf},
                             {2.40e+2F, 0.F, 248.F, hf},
                             {256.F, 0.F, 264.F, hf},
                             {272.F, 0.F, 2.80e+2F, hf},
                             {288.F, 0.F, 296.F, hf},
                             {304.F, 0.F, 312.F, hf},
                             {3.20e+2F, 0.F, 328.F, hf},
                             {336.F, 0.F, 344.F, hf},
                             {352.F, 0.F, 3.60e+2F, hf},
                             {368.F, 0.F, 376.F, hf},
                             {384.F, 0.F, 392.F, hf},
                             {4.00e+2F, 0.F, 408.F, hf},
                             {416.F, 0.F, 424.F, hf},
                             {432.F, 0.F, 4.40e+2F, hf},
                             {448.F, 0.F, 456.F, hf},
                             {464.F, 0.F, 472.F, hf},
                             {4.80e+2F, 0.F, 488.F, hf},
                             {496.F, 0.F, 504.F, hf},
                             {512.F, 0.F, 5.20e+2F, hf},
                             {528.F, 0.F, 536.F, hf},
                             {544.F, 0.F, 552.F, hf},
                             {5.60e+2F, 0.F, 568.F, hf},
                             {576.F, 0.F, 584.F, hf},
                             {592.F, 0.F, 6.00e+2F, hf},
                             {608.F, 0.F, 616.F, hf},
                             {624.F, 0.F, 632.F, hf},
                             {6.40e+2F, 0.F, 648.F, hf},
                             {656.F, 0.F, 664.F, hf},
                             {672.F, 0.F, 6.80e+2F, hf},
                             {688.F, 0.F, 696.F, hf},
                             {704.F, 0.F, 712.F, hf},
                             {7.20e+2F, 0.F, 728.F, hf},
                             {736.F, 0.F, 744.F, hf},
                             {752.F, 0.F, 7.60e+2F, hf},
                             {768.F, 0.F, 776.F, hf},
                             {784.F, 0.F, 792.F, hf},
                             {8.00e+2F, 0.F, 808.F, hf},
                             {816.F, 0.F, 824.F, hf},
                             {832.F, 0.F, 8.40e+2F, hf},
                             {848.F, 0.F, 856.F, hf},
                             {864.F, 0.F, 872.F, hf},
                             {8.80e+2F, 0.F, 888.F, hf},
                             {896.F, 0.F, 904.F, hf},
                             {912.F, 0.F, 9.20e+2F, hf},
                             {928.F, 0.F, 936.F, hf},
                             {944.F, 0.F, 952.F, hf},
                             {9.60e+2F, 0.F, 968.F, hf},
                             {976.F, 0.F, 984.F, hf},
                             {992.F, 0.F, 1.00e+3F, hf},
                             {1008.F, 0.F, 1016.F, hf},
                             {1024.F, 0.F, 1032.F, hf},
                             {1.040e+3F, 0.F, 1048.F, hf},
                             {1056.F, 0.F, 1064.F, hf},
                             {1072.F, 0.F, 1.080e+3F, hf},
                             {1088.F, 0.F, 1096.F, hf},
                             {1104.F, 0.F, 1112.F, hf},
                             {1.120e+3F, 0.F, 1128.F, hf},
                             {1136.F, 0.F, 1144.F, hf},
                             {1152.F, 0.F, 1.160e+3F, hf},
                             {1168.F, 0.F, 1176.F, hf},
                             {1184.F, 0.F, 1192.F, hf},
                             {1.20e+3F, 0.F, 1208.F, hf},
                             {1216.F, 0.F, 1224.F, hf},
                             {1232.F, 0.F, 1.240e+3F, hf},
                             {1248.F, 0.F, 1256.F, hf},
                             {1264.F, 0.F, 1272.F, hf},
                             {1.280e+3F, 0.F, 1288.F, hf},
                             {1296.F, 0.F, 1304.F, hf},
                             {1312.F, 0.F, 1.320e+3F, hf},
                             {1328.F, 0.F, 1336.F, hf},
                             {1344.F, 0.F, 1352.F, hf},
                             {1.360e+3F, 0.F, 1368.F, hf},
                             {1376.F, 0.F, 1384.F, hf},
                             {1392.F, 0.F, 1.40e+3F, hf},
                             {1408.F, 0.F, 1416.F, hf},
                             {1424.F, 0.F, 1432.F, hf},
                             {1.440e+3F, 0.F, 1448.F, hf},
                             {1456.F, 0.F, 1464.F, hf},
                             {1472.F, 0.F, 1.480e+3F, hf},
                             {1488.F, 0.F, 1496.F, hf},
                             {1504.F, 0.F, 1512.F, hf},
                             {1.520e+3F, 0.F, 1528.F, hf},
                             {1536.F, 0.F, 1544.F, hf},
                             {1552.F, 0.F, 1.560e+3F, hf},
                             {1568.F, 0.F, 1576.F, hf},
                             {1584.F, 0.F, 1592.F, hf},
                             {1.60e+3F, 0.F, 1608.F, hf},
                             {1616.F, 0.F, 1624.F, hf},
                             {1632.F, 0.F, 1.640e+3F, hf},
                             {1648.F, 0.F, 1656.F, hf},
                             {1664.F, 0.F, 1672.F, hf},
                             {1.680e+3F, 0.F, 1688.F, hf},
                             {1696.F, 0.F, 1704.F, hf},
                             {1712.F, 0.F, 1.720e+3F, hf},
                             {1728.F, 0.F, 1736.F, hf},
                             {1744.F, 0.F, 1752.F, hf},
                             {1.760e+3F, 0.F, 1768.F, hf},
                             {1776.F, 0.F, 1784.F, hf},
                             {1792.F, 0.F, 1.80e+3F, hf},
                             {1808.F, 0.F, 1816.F, hf},
                             {1824.F, 0.F, 1832.F, hf},
                             {1.840e+3F, 0.F, 1848.F, hf},
                             {1856.F, 0.F, 1864.F, hf},
                             {1872.F, 0.F, 1.880e+3F, hf},
                             {1888.F, 0.F, 1896.F, hf},
                             {1904.F, 0.F, 1912.F, hf},
                             {1.920e+3F, 0.F, 1928.F, hf},
                             {1936.F, 0.F, 1944.F, hf},
                             {1952.F, 0.F, 1.960e+3F, hf},
                             {1968.F, 0.F, 1976.F, hf},
                             {1984.F, 0.F, 1992.F, hf},
                             {2.00e+3F, 0.F, 2008.F, hf},
                             {2016.F, 0.F, 2024.F, hf},
                             {2032.F, 0.F, 2.040e+3F, hf}}}}},
       {.id = 17,
        .name = "vertical-stripes-7-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, 8.0F, hf},
                             {16.F, 0.F, 24.F, hf},
                             {32.F, 0.F, 40.F, hf},
                             {48.F, 0.F, 56.F, hf},
                             {64.F, 0.F, 72.F, hf},
                             {80.F, 0.F, 88.F, hf},
                             {96.F, 0.F, 104.F, hf},
                             {112.F, 0.F, 1.20e+2F, hf},
                             {128.F, 0.F, 136.F, hf},
                             {144.F, 0.F, 152.F, hf},
                             {1.60e+2F, 0.F, 168.F, hf},
                             {176.F, 0.F, 184.F, hf},
                             {192.F, 0.F, 2.00e+2F, hf},
                             {208.F, 0.F, 216.F, hf},
                             {224.F, 0.F, 232.F, hf},
                             {2.40e+2F, 0.F, 248.F, hf},
                             {256.F, 0.F, 264.F, hf},
                             {272.F, 0.F, 2.80e+2F, hf},
                             {288.F, 0.F, 296.F, hf},
                             {304.F, 0.F, 312.F, hf},
                             {3.20e+2F, 0.F, 328.F, hf},
                             {336.F, 0.F, 344.F, hf},
                             {352.F, 0.F, 3.60e+2F, hf},
                             {368.F, 0.F, 376.F, hf},
                             {384.F, 0.F, 392.F, hf},
                             {4.00e+2F, 0.F, 408.F, hf},
                             {416.F, 0.F, 424.F, hf},
                             {432.F, 0.F, 4.40e+2F, hf},
                             {448.F, 0.F, 456.F, hf},
                             {464.F, 0.F, 472.F, hf},
                             {4.80e+2F, 0.F, 488.F, hf},
                             {496.F, 0.F, 504.F, hf},
                             {512.F, 0.F, 5.20e+2F, hf},
                             {528.F, 0.F, 536.F, hf},
                             {544.F, 0.F, 552.F, hf},
                             {5.60e+2F, 0.F, 568.F, hf},
                             {576.F, 0.F, 584.F, hf},
                             {592.F, 0.F, 6.00e+2F, hf},
                             {608.F, 0.F, 616.F, hf},
                             {624.F, 0.F, 632.F, hf},
                             {6.40e+2F, 0.F, 648.F, hf},
                             {656.F, 0.F, 664.F, hf},
                             {672.F, 0.F, 6.80e+2F, hf},
                             {688.F, 0.F, 696.F, hf},
                             {704.F, 0.F, 712.F, hf},
                             {7.20e+2F, 0.F, 728.F, hf},
                             {736.F, 0.F, 744.F, hf},
                             {752.F, 0.F, 7.60e+2F, hf},
                             {768.F, 0.F, 776.F, hf},
                             {784.F, 0.F, 792.F, hf},
                             {8.00e+2F, 0.F, 808.F, hf},
                             {816.F, 0.F, 824.F, hf},
                             {832.F, 0.F, 8.40e+2F, hf},
                             {848.F, 0.F, 856.F, hf},
                             {864.F, 0.F, 872.F, hf},
                             {8.80e+2F, 0.F, 888.F, hf},
                             {896.F, 0.F, 904.F, hf},
                             {912.F, 0.F, 9.20e+2F, hf},
                             {928.F, 0.F, 936.F, hf},
                             {944.F, 0.F, 952.F, hf},
                             {9.60e+2F, 0.F, 968.F, hf},
                             {976.F, 0.F, 984.F, hf},
                             {992.F, 0.F, 1.00e+3F, hf},
                             {1008.F, 0.F, 1016.F, hf},
                             {1024.F, 0.F, 1032.F, hf},
                             {1.040e+3F, 0.F, 1048.F, hf},
                             {1056.F, 0.F, 1064.F, hf},
                             {1072.F, 0.F, 1.080e+3F, hf},
                             {1088.F, 0.F, 1096.F, hf},
                             {1104.F, 0.F, 1112.F, hf},
                             {1.120e+3F, 0.F, 1128.F, hf},
                             {1136.F, 0.F, 1144.F, hf},
                             {1152.F, 0.F, 1.160e+3F, hf},
                             {1168.F, 0.F, 1176.F, hf},
                             {1184.F, 0.F, 1192.F, hf},
                             {1.20e+3F, 0.F, 1208.F, hf},
                             {1216.F, 0.F, 1224.F, hf},
                             {1232.F, 0.F, 1.240e+3F, hf},
                             {1248.F, 0.F, 1256.F, hf},
                             {1264.F, 0.F, 1272.F, hf},
                             {1.280e+3F, 0.F, 1288.F, hf},
                             {1296.F, 0.F, 1304.F, hf},
                             {1312.F, 0.F, 1.320e+3F, hf},
                             {1328.F, 0.F, 1336.F, hf},
                             {1344.F, 0.F, 1352.F, hf},
                             {1.360e+3F, 0.F, 1368.F, hf},
                             {1376.F, 0.F, 1384.F, hf},
                             {1392.F, 0.F, 1.40e+3F, hf},
                             {1408.F, 0.F, 1416.F, hf},
                             {1424.F, 0.F, 1432.F, hf},
                             {1.440e+3F, 0.F, 1448.F, hf},
                             {1456.F, 0.F, 1464.F, hf},
                             {1472.F, 0.F, 1.480e+3F, hf},
                             {1488.F, 0.F, 1496.F, hf},
                             {1504.F, 0.F, 1512.F, hf},
                             {1.520e+3F, 0.F, 1528.F, hf},
                             {1536.F, 0.F, 1544.F, hf},
                             {1552.F, 0.F, 1.560e+3F, hf},
                             {1568.F, 0.F, 1576.F, hf},
                             {1584.F, 0.F, 1592.F, hf},
                             {1.60e+3F, 0.F, 1608.F, hf},
                             {1616.F, 0.F, 1624.F, hf},
                             {1632.F, 0.F, 1.640e+3F, hf},
                             {1648.F, 0.F, 1656.F, hf},
                             {1664.F, 0.F, 1672.F, hf},
                             {1.680e+3F, 0.F, 1688.F, hf},
                             {1696.F, 0.F, 1704.F, hf},
                             {1712.F, 0.F, 1.720e+3F, hf},
                             {1728.F, 0.F, 1736.F, hf},
                             {1744.F, 0.F, 1752.F, hf},
                             {1.760e+3F, 0.F, 1768.F, hf},
                             {1776.F, 0.F, 1784.F, hf},
                             {1792.F, 0.F, 1.80e+3F, hf},
                             {1808.F, 0.F, 1816.F, hf},
                             {1824.F, 0.F, 1832.F, hf},
                             {1.840e+3F, 0.F, 1848.F, hf},
                             {1856.F, 0.F, 1864.F, hf},
                             {1872.F, 0.F, 1.880e+3F, hf},
                             {1888.F, 0.F, 1896.F, hf},
                             {1904.F, 0.F, 1912.F, hf},
                             {1.920e+3F, 0.F, 1928.F, hf},
                             {1936.F, 0.F, 1944.F, hf},
                             {1952.F, 0.F, 1.960e+3F, hf},
                             {1968.F, 0.F, 1976.F, hf},
                             {1984.F, 0.F, 1992.F, hf},
                             {2.00e+3F, 0.F, 2008.F, hf},
                             {2016.F, 0.F, 2024.F, hf},
                             {2032.F, 0.F, 2.040e+3F, hf}}}}},
       {.id = 18,
        .name = "horizontal-stripes-0-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, 1024.F}}}}},
       {.id = 19,
        .name = "horizontal-stripes-0-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, 1024.F}}}}},
       {.id = 20,
        .name = "horizontal-stripes-1-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, 512.F},
                             {0.F, 1024.F, wf, 1536.F}}}}},
       {.id = 21,
        .name = "horizontal-stripes-1-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, 512.F},
                             {0.F, 1024.F, wf, 1536.F}}}}},
       {.id = 22,
        .name = "horizontal-stripes-2-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, 256.F},
                             {0.F, 512.F, wf, 768.F},
                             {0.F, 1024.F, wf, 1.280e+3F},
                             {0.F, 1536.F, wf, 1792.F}}}}},
       {.id = 23,
        .name = "horizontal-stripes-2-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, 256.F},
                             {0.F, 512.F, wf, 768.F},
                             {0.F, 1024.F, wf, 1.280e+3F},
                             {0.F, 1536.F, wf, 1792.F}}}}},
       {.id = 24,
        .name = "horizontal-stripes-3-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, 128.F},
                             {0.F, 256.F, wf, 384.F},
                             {0.F, 512.F, wf, 6.40e+2F},
                             {0.F, 768.F, wf, 896.F},
                             {0.F, 1024.F, wf, 1152.F},
                             {0.F, 1.280e+3F, wf, 1408.F},
                             {0.F, 1536.F, wf, 1664.F},
                             {0.F, 1792.F, wf, 1.920e+3F}}}}},
       {.id = 25,
        .name = "horizontal-stripes-3-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, 128.F},
                             {0.F, 256.F, wf, 384.F},
                             {0.F, 512.F, wf, 6.40e+2F},
                             {0.F, 768.F, wf, 896.F},
                             {0.F, 1024.F, wf, 1152.F},
                             {0.F, 1.280e+3F, wf, 1408.F},
                             {0.F, 1536.F, wf, 1664.F},
                             {0.F, 1792.F, wf, 1.920e+3F}}}}},
       {.id = 26,
        .name = "horizontal-stripes-4-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, 64.F},
                             {0.F, 128.F, wf, 192.F},
                             {0.F, 256.F, wf, 3.20e+2F},
                             {0.F, 384.F, wf, 448.F},
                             {0.F, 512.F, wf, 576.F},
                             {0.F, 6.40e+2F, wf, 704.F},
                             {0.F, 768.F, wf, 832.F},
                             {0.F, 896.F, wf, 9.60e+2F},
                             {0.F, 1024.F, wf, 1088.F},
                             {0.F, 1152.F, wf, 1216.F},
                             {0.F, 1.280e+3F, wf, 1344.F},
                             {0.F, 1408.F, wf, 1472.F},
                             {0.F, 1536.F, wf, 1.60e+3F},
                             {0.F, 1664.F, wf, 1728.F},
                             {0.F, 1792.F, wf, 1856.F},
                             {0.F, 1.920e+3F, wf, 1984.F}}}}},
       {.id = 27,
        .name = "horizontal-stripes-4-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, 64.F},
                             {0.F, 128.F, wf, 192.F},
                             {0.F, 256.F, wf, 3.20e+2F},
                             {0.F, 384.F, wf, 448.F},
                             {0.F, 512.F, wf, 576.F},
                             {0.F, 6.40e+2F, wf, 704.F},
                             {0.F, 768.F, wf, 832.F},
                             {0.F, 896.F, wf, 9.60e+2F},
                             {0.F, 1024.F, wf, 1088.F},
                             {0.F, 1152.F, wf, 1216.F},
                             {0.F, 1.280e+3F, wf, 1344.F},
                             {0.F, 1408.F, wf, 1472.F},
                             {0.F, 1536.F, wf, 1.60e+3F},
                             {0.F, 1664.F, wf, 1728.F},
                             {0.F, 1792.F, wf, 1856.F},
                             {0.F, 1.920e+3F, wf, 1984.F}}}}},
       {.id = 28,
        .name = "horizontal-stripes-5-normal",
        .draw =
            {{.color = {dark, dark, dark},
              .type =
                  GL_QUADS,
              .coords = {{0.F, 0.F, wf, hf}}},
             {.color = {bright, bright, bright},
              .type =
                  GL_QUADS,
              .coords =
                  {{0.F, 0.F, wf, 32.F},         {0.F, 64.F, wf, 96.F},
                   {0.F, 128.F, wf, 1.60e+2F},   {0.F, 192.F, wf, 224.F},
                   {0.F, 256.F, wf, 288.F},      {0.F, 3.20e+2F, wf, 352.F},
                   {0.F, 384.F, wf, 416.F},      {0.F, 448.F, wf, 4.80e+2F},
                   {0.F, 512.F, wf, 544.F},      {0.F, 576.F, wf, 608.F},
                   {0.F, 6.40e+2F, wf, 672.F},   {0.F, 704.F, wf, 736.F},
                   {0.F, 768.F, wf, 8.00e+2F},   {0.F, 832.F, wf, 864.F},
                   {0.F, 896.F, wf, 928.F},      {0.F, 9.60e+2F, wf, 992.F},
                   {0.F, 1024.F, wf, 1056.F},    {0.F, 1088.F, wf, 1.120e+3F},
                   {0.F, 1152.F, wf, 1184.F},    {0.F, 1216.F, wf, 1248.F},
                   {0.F, 1.280e+3F, wf, 1312.F}, {0.F, 1344.F, wf, 1376.F},
                   {0.F, 1408.F, wf, 1.440e+3F}, {0.F, 1472.F, wf, 1504.F},
                   {0.F, 1536.F, wf, 1568.F},    {0.F, 1.60e+3F, wf, 1632.F},
                   {0.F, 1664.F, wf, 1696.F},    {0.F, 1728.F, wf, 1.760e+3F},
                   {0.F, 1792.F, wf, 1824.F},    {0.F, 1856.F, wf, 1888.F},
                   {0.F, 1.920e+3F, wf, 1952.F}, {0.F, 1984.F, wf, 2016.F}}}}},
       {.id = 29,
        .name = "horizontal-stripes-5-invert",
        .draw =
            {{.color = {bright, bright, bright},
              .type =
                  GL_QUADS,
              .coords = {{0.F, 0.F, wf, hf}}},
             {.color = {dark, dark, dark},
              .type =
                  GL_QUADS,
              .coords =
                  {{0.F, 0.F, wf, 32.F},         {0.F, 64.F, wf, 96.F},
                   {0.F, 128.F, wf, 1.60e+2F},   {0.F, 192.F, wf, 224.F},
                   {0.F, 256.F, wf, 288.F},      {0.F, 3.20e+2F, wf, 352.F},
                   {0.F, 384.F, wf, 416.F},      {0.F, 448.F, wf, 4.80e+2F},
                   {0.F, 512.F, wf, 544.F},      {0.F, 576.F, wf, 608.F},
                   {0.F, 6.40e+2F, wf, 672.F},   {0.F, 704.F, wf, 736.F},
                   {0.F, 768.F, wf, 8.00e+2F},   {0.F, 832.F, wf, 864.F},
                   {0.F, 896.F, wf, 928.F},      {0.F, 9.60e+2F, wf, 992.F},
                   {0.F, 1024.F, wf, 1056.F},    {0.F, 1088.F, wf, 1.120e+3F},
                   {0.F, 1152.F, wf, 1184.F},    {0.F, 1216.F, wf, 1248.F},
                   {0.F, 1.280e+3F, wf, 1312.F}, {0.F, 1344.F, wf, 1376.F},
                   {0.F, 1408.F, wf, 1.440e+3F}, {0.F, 1472.F, wf, 1504.F},
                   {0.F, 1536.F, wf, 1568.F},    {0.F, 1.60e+3F, wf, 1632.F},
                   {0.F, 1664.F, wf, 1696.F},    {0.F, 1728.F, wf, 1.760e+3F},
                   {0.F, 1792.F, wf, 1824.F},    {0.F, 1856.F, wf, 1888.F},
                   {0.F, 1.920e+3F, wf, 1952.F}, {0.F, 1984.F, wf, 2016.F}}}}},
       {.id = 30,
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
                  {{0.F, 0.F, wf, 16.F},         {0.F, 32.F, wf, 48.F},
                   {0.F, 64.F, wf, 80.F},        {0.F, 96.F, wf, 112.F},
                   {0.F, 128.F, wf, 144.F},      {0.F, 1.60e+2F, wf, 176.F},
                   {0.F, 192.F, wf, 208.F},      {0.F, 224.F, wf, 2.40e+2F},
                   {0.F, 256.F, wf, 272.F},      {0.F, 288.F, wf, 304.F},
                   {0.F, 3.20e+2F, wf, 336.F},   {0.F, 352.F, wf, 368.F},
                   {0.F, 384.F, wf, 4.00e+2F},   {0.F, 416.F, wf, 432.F},
                   {0.F, 448.F, wf, 464.F},      {0.F, 4.80e+2F, wf, 496.F},
                   {0.F, 512.F, wf, 528.F},      {0.F, 544.F, wf, 5.60e+2F},
                   {0.F, 576.F, wf, 592.F},      {0.F, 608.F, wf, 624.F},
                   {0.F, 6.40e+2F, wf, 656.F},   {0.F, 672.F, wf, 688.F},
                   {0.F, 704.F, wf, 7.20e+2F},   {0.F, 736.F, wf, 752.F},
                   {0.F, 768.F, wf, 784.F},      {0.F, 8.00e+2F, wf, 816.F},
                   {0.F, 832.F, wf, 848.F},      {0.F, 864.F, wf, 8.80e+2F},
                   {0.F, 896.F, wf, 912.F},      {0.F, 928.F, wf, 944.F},
                   {0.F, 9.60e+2F, wf, 976.F},   {0.F, 992.F, wf, 1008.F},
                   {0.F, 1024.F, wf, 1.040e+3F}, {0.F, 1056.F, wf, 1072.F},
                   {0.F, 1088.F, wf, 1104.F},    {0.F, 1.120e+3F, wf, 1136.F},
                   {0.F, 1152.F, wf, 1168.F},    {0.F, 1184.F, wf, 1.20e+3F},
                   {0.F, 1216.F, wf, 1232.F},    {0.F, 1248.F, wf, 1264.F},
                   {0.F, 1.280e+3F, wf, 1296.F}, {0.F, 1312.F, wf, 1328.F},
                   {0.F, 1344.F, wf, 1.360e+3F}, {0.F, 1376.F, wf, 1392.F},
                   {0.F, 1408.F, wf, 1424.F},    {0.F, 1.440e+3F, wf, 1456.F},
                   {0.F, 1472.F, wf, 1488.F},    {0.F, 1504.F, wf, 1.520e+3F},
                   {0.F, 1536.F, wf, 1552.F},    {0.F, 1568.F, wf, 1584.F},
                   {0.F, 1.60e+3F, wf, 1616.F},  {0.F, 1632.F, wf, 1648.F},
                   {0.F, 1664.F, wf, 1.680e+3F}, {0.F, 1696.F, wf, 1712.F},
                   {0.F, 1728.F, wf, 1744.F},    {0.F, 1.760e+3F, wf, 1776.F},
                   {0.F, 1792.F, wf, 1808.F},    {0.F, 1824.F, wf, 1.840e+3F},
                   {0.F, 1856.F, wf, 1872.F},    {0.F, 1888.F, wf, 1904.F},
                   {0.F, 1.920e+3F, wf, 1936.F}, {0.F, 1952.F, wf, 1968.F},
                   {0.F, 1984.F, wf, 2.00e+3F},  {0.F, 2016.F, wf, 2032.F}}}}},
       {.id = 31,
        .name = "horizontal-stripes-6-invert",
        .draw =
            {{.color = {bright, bright, bright},
              .type =
                  GL_QUADS,
              .coords = {{0.F, 0.F, wf, hf}}},
             {.color = {dark, dark, dark},
              .type =
                  GL_QUADS,
              .coords =
                  {{0.F, 0.F, wf, 16.F},         {0.F, 32.F, wf, 48.F},
                   {0.F, 64.F, wf, 80.F},        {0.F, 96.F, wf, 112.F},
                   {0.F, 128.F, wf, 144.F},      {0.F, 1.60e+2F, wf, 176.F},
                   {0.F, 192.F, wf, 208.F},      {0.F, 224.F, wf, 2.40e+2F},
                   {0.F, 256.F, wf, 272.F},      {0.F, 288.F, wf, 304.F},
                   {0.F, 3.20e+2F, wf, 336.F},   {0.F, 352.F, wf, 368.F},
                   {0.F, 384.F, wf, 4.00e+2F},   {0.F, 416.F, wf, 432.F},
                   {0.F, 448.F, wf, 464.F},      {0.F, 4.80e+2F, wf, 496.F},
                   {0.F, 512.F, wf, 528.F},      {0.F, 544.F, wf, 5.60e+2F},
                   {0.F, 576.F, wf, 592.F},      {0.F, 608.F, wf, 624.F},
                   {0.F, 6.40e+2F, wf, 656.F},   {0.F, 672.F, wf, 688.F},
                   {0.F, 704.F, wf, 7.20e+2F},   {0.F, 736.F, wf, 752.F},
                   {0.F, 768.F, wf, 784.F},      {0.F, 8.00e+2F, wf, 816.F},
                   {0.F, 832.F, wf, 848.F},      {0.F, 864.F, wf, 8.80e+2F},
                   {0.F, 896.F, wf, 912.F},      {0.F, 928.F, wf, 944.F},
                   {0.F, 9.60e+2F, wf, 976.F},   {0.F, 992.F, wf, 1008.F},
                   {0.F, 1024.F, wf, 1.040e+3F}, {0.F, 1056.F, wf, 1072.F},
                   {0.F, 1088.F, wf, 1104.F},    {0.F, 1.120e+3F, wf, 1136.F},
                   {0.F, 1152.F, wf, 1168.F},    {0.F, 1184.F, wf, 1.20e+3F},
                   {0.F, 1216.F, wf, 1232.F},    {0.F, 1248.F, wf, 1264.F},
                   {0.F, 1.280e+3F, wf, 1296.F}, {0.F, 1312.F, wf, 1328.F},
                   {0.F, 1344.F, wf, 1.360e+3F}, {0.F, 1376.F, wf, 1392.F},
                   {0.F, 1408.F, wf, 1424.F},    {0.F, 1.440e+3F, wf, 1456.F},
                   {0.F, 1472.F, wf, 1488.F},    {0.F, 1504.F, wf, 1.520e+3F},
                   {0.F, 1536.F, wf, 1552.F},    {0.F, 1568.F, wf, 1584.F},
                   {0.F, 1.60e+3F, wf, 1616.F},  {0.F, 1632.F, wf, 1648.F},
                   {0.F, 1664.F, wf, 1.680e+3F}, {0.F, 1696.F, wf, 1712.F},
                   {0.F, 1728.F, wf, 1744.F},    {0.F, 1.760e+3F, wf, 1776.F},
                   {0.F, 1792.F, wf, 1808.F},    {0.F, 1824.F, wf, 1.840e+3F},
                   {0.F, 1856.F, wf, 1872.F},    {0.F, 1888.F, wf, 1904.F},
                   {0.F, 1.920e+3F, wf, 1936.F}, {0.F, 1952.F, wf, 1968.F},
                   {0.F, 1984.F, wf, 2.00e+3F},  {0.F, 2016.F, wf, 2032.F}}}}},
       {.id = 32,
        .name = "horizontal-stripes-7-normal",
        .draw = {{.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, 8.0F},
                             {0.F, 16.F, wf, 24.F},
                             {0.F, 32.F, wf, 40.F},
                             {0.F, 48.F, wf, 56.F},
                             {0.F, 64.F, wf, 72.F},
                             {0.F, 80.F, wf, 88.F},
                             {0.F, 96.F, wf, 104.F},
                             {0.F, 112.F, wf, 1.20e+2F},
                             {0.F, 128.F, wf, 136.F},
                             {0.F, 144.F, wf, 152.F},
                             {0.F, 1.60e+2F, wf, 168.F},
                             {0.F, 176.F, wf, 184.F},
                             {0.F, 192.F, wf, 2.00e+2F},
                             {0.F, 208.F, wf, 216.F},
                             {0.F, 224.F, wf, 232.F},
                             {0.F, 2.40e+2F, wf, 248.F},
                             {0.F, 256.F, wf, 264.F},
                             {0.F, 272.F, wf, 2.80e+2F},
                             {0.F, 288.F, wf, 296.F},
                             {0.F, 304.F, wf, 312.F},
                             {0.F, 3.20e+2F, wf, 328.F},
                             {0.F, 336.F, wf, 344.F},
                             {0.F, 352.F, wf, 3.60e+2F},
                             {0.F, 368.F, wf, 376.F},
                             {0.F, 384.F, wf, 392.F},
                             {0.F, 4.00e+2F, wf, 408.F},
                             {0.F, 416.F, wf, 424.F},
                             {0.F, 432.F, wf, 4.40e+2F},
                             {0.F, 448.F, wf, 456.F},
                             {0.F, 464.F, wf, 472.F},
                             {0.F, 4.80e+2F, wf, 488.F},
                             {0.F, 496.F, wf, 504.F},
                             {0.F, 512.F, wf, 5.20e+2F},
                             {0.F, 528.F, wf, 536.F},
                             {0.F, 544.F, wf, 552.F},
                             {0.F, 5.60e+2F, wf, 568.F},
                             {0.F, 576.F, wf, 584.F},
                             {0.F, 592.F, wf, 6.00e+2F},
                             {0.F, 608.F, wf, 616.F},
                             {0.F, 624.F, wf, 632.F},
                             {0.F, 6.40e+2F, wf, 648.F},
                             {0.F, 656.F, wf, 664.F},
                             {0.F, 672.F, wf, 6.80e+2F},
                             {0.F, 688.F, wf, 696.F},
                             {0.F, 704.F, wf, 712.F},
                             {0.F, 7.20e+2F, wf, 728.F},
                             {0.F, 736.F, wf, 744.F},
                             {0.F, 752.F, wf, 7.60e+2F},
                             {0.F, 768.F, wf, 776.F},
                             {0.F, 784.F, wf, 792.F},
                             {0.F, 8.00e+2F, wf, 808.F},
                             {0.F, 816.F, wf, 824.F},
                             {0.F, 832.F, wf, 8.40e+2F},
                             {0.F, 848.F, wf, 856.F},
                             {0.F, 864.F, wf, 872.F},
                             {0.F, 8.80e+2F, wf, 888.F},
                             {0.F, 896.F, wf, 904.F},
                             {0.F, 912.F, wf, 9.20e+2F},
                             {0.F, 928.F, wf, 936.F},
                             {0.F, 944.F, wf, 952.F},
                             {0.F, 9.60e+2F, wf, 968.F},
                             {0.F, 976.F, wf, 984.F},
                             {0.F, 992.F, wf, 1.00e+3F},
                             {0.F, 1008.F, wf, 1016.F},
                             {0.F, 1024.F, wf, 1032.F},
                             {0.F, 1.040e+3F, wf, 1048.F},
                             {0.F, 1056.F, wf, 1064.F},
                             {0.F, 1072.F, wf, 1.080e+3F},
                             {0.F, 1088.F, wf, 1096.F},
                             {0.F, 1104.F, wf, 1112.F},
                             {0.F, 1.120e+3F, wf, 1128.F},
                             {0.F, 1136.F, wf, 1144.F},
                             {0.F, 1152.F, wf, 1.160e+3F},
                             {0.F, 1168.F, wf, 1176.F},
                             {0.F, 1184.F, wf, 1192.F},
                             {0.F, 1.20e+3F, wf, 1208.F},
                             {0.F, 1216.F, wf, 1224.F},
                             {0.F, 1232.F, wf, 1.240e+3F},
                             {0.F, 1248.F, wf, 1256.F},
                             {0.F, 1264.F, wf, 1272.F},
                             {0.F, 1.280e+3F, wf, 1288.F},
                             {0.F, 1296.F, wf, 1304.F},
                             {0.F, 1312.F, wf, 1.320e+3F},
                             {0.F, 1328.F, wf, 1336.F},
                             {0.F, 1344.F, wf, 1352.F},
                             {0.F, 1.360e+3F, wf, 1368.F},
                             {0.F, 1376.F, wf, 1384.F},
                             {0.F, 1392.F, wf, 1.40e+3F},
                             {0.F, 1408.F, wf, 1416.F},
                             {0.F, 1424.F, wf, 1432.F},
                             {0.F, 1.440e+3F, wf, 1448.F},
                             {0.F, 1456.F, wf, 1464.F},
                             {0.F, 1472.F, wf, 1.480e+3F},
                             {0.F, 1488.F, wf, 1496.F},
                             {0.F, 1504.F, wf, 1512.F},
                             {0.F, 1.520e+3F, wf, 1528.F},
                             {0.F, 1536.F, wf, 1544.F},
                             {0.F, 1552.F, wf, 1.560e+3F},
                             {0.F, 1568.F, wf, 1576.F},
                             {0.F, 1584.F, wf, 1592.F},
                             {0.F, 1.60e+3F, wf, 1608.F},
                             {0.F, 1616.F, wf, 1624.F},
                             {0.F, 1632.F, wf, 1.640e+3F},
                             {0.F, 1648.F, wf, 1656.F},
                             {0.F, 1664.F, wf, 1672.F},
                             {0.F, 1.680e+3F, wf, 1688.F},
                             {0.F, 1696.F, wf, 1704.F},
                             {0.F, 1712.F, wf, 1.720e+3F},
                             {0.F, 1728.F, wf, 1736.F},
                             {0.F, 1744.F, wf, 1752.F},
                             {0.F, 1.760e+3F, wf, 1768.F},
                             {0.F, 1776.F, wf, 1784.F},
                             {0.F, 1792.F, wf, 1.80e+3F},
                             {0.F, 1808.F, wf, 1816.F},
                             {0.F, 1824.F, wf, 1832.F},
                             {0.F, 1.840e+3F, wf, 1848.F},
                             {0.F, 1856.F, wf, 1864.F},
                             {0.F, 1872.F, wf, 1.880e+3F},
                             {0.F, 1888.F, wf, 1896.F},
                             {0.F, 1904.F, wf, 1912.F},
                             {0.F, 1.920e+3F, wf, 1928.F},
                             {0.F, 1936.F, wf, 1944.F},
                             {0.F, 1952.F, wf, 1.960e+3F},
                             {0.F, 1968.F, wf, 1976.F},
                             {0.F, 1984.F, wf, 1992.F},
                             {0.F, 2.00e+3F, wf, 2008.F},
                             {0.F, 2016.F, wf, 2024.F},
                             {0.F, 2032.F, wf, 2.040e+3F}}}}},
       {.id = 33,
        .name = "horizontal-stripes-7-invert",
        .draw = {{.color = {bright, bright, bright},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, hf}}},
                 {.color = {dark, dark, dark},
                  .type = GL_QUADS,
                  .coords = {{0.F, 0.F, wf, 8.0F},
                             {0.F, 16.F, wf, 24.F},
                             {0.F, 32.F, wf, 40.F},
                             {0.F, 48.F, wf, 56.F},
                             {0.F, 64.F, wf, 72.F},
                             {0.F, 80.F, wf, 88.F},
                             {0.F, 96.F, wf, 104.F},
                             {0.F, 112.F, wf, 1.20e+2F},
                             {0.F, 128.F, wf, 136.F},
                             {0.F, 144.F, wf, 152.F},
                             {0.F, 1.60e+2F, wf, 168.F},
                             {0.F, 176.F, wf, 184.F},
                             {0.F, 192.F, wf, 2.00e+2F},
                             {0.F, 208.F, wf, 216.F},
                             {0.F, 224.F, wf, 232.F},
                             {0.F, 2.40e+2F, wf, 248.F},
                             {0.F, 256.F, wf, 264.F},
                             {0.F, 272.F, wf, 2.80e+2F},
                             {0.F, 288.F, wf, 296.F},
                             {0.F, 304.F, wf, 312.F},
                             {0.F, 3.20e+2F, wf, 328.F},
                             {0.F, 336.F, wf, 344.F},
                             {0.F, 352.F, wf, 3.60e+2F},
                             {0.F, 368.F, wf, 376.F},
                             {0.F, 384.F, wf, 392.F},
                             {0.F, 4.00e+2F, wf, 408.F},
                             {0.F, 416.F, wf, 424.F},
                             {0.F, 432.F, wf, 4.40e+2F},
                             {0.F, 448.F, wf, 456.F},
                             {0.F, 464.F, wf, 472.F},
                             {0.F, 4.80e+2F, wf, 488.F},
                             {0.F, 496.F, wf, 504.F},
                             {0.F, 512.F, wf, 5.20e+2F},
                             {0.F, 528.F, wf, 536.F},
                             {0.F, 544.F, wf, 552.F},
                             {0.F, 5.60e+2F, wf, 568.F},
                             {0.F, 576.F, wf, 584.F},
                             {0.F, 592.F, wf, 6.00e+2F},
                             {0.F, 608.F, wf, 616.F},
                             {0.F, 624.F, wf, 632.F},
                             {0.F, 6.40e+2F, wf, 648.F},
                             {0.F, 656.F, wf, 664.F},
                             {0.F, 672.F, wf, 6.80e+2F},
                             {0.F, 688.F, wf, 696.F},
                             {0.F, 704.F, wf, 712.F},
                             {0.F, 7.20e+2F, wf, 728.F},
                             {0.F, 736.F, wf, 744.F},
                             {0.F, 752.F, wf, 7.60e+2F},
                             {0.F, 768.F, wf, 776.F},
                             {0.F, 784.F, wf, 792.F},
                             {0.F, 8.00e+2F, wf, 808.F},
                             {0.F, 816.F, wf, 824.F},
                             {0.F, 832.F, wf, 8.40e+2F},
                             {0.F, 848.F, wf, 856.F},
                             {0.F, 864.F, wf, 872.F},
                             {0.F, 8.80e+2F, wf, 888.F},
                             {0.F, 896.F, wf, 904.F},
                             {0.F, 912.F, wf, 9.20e+2F},
                             {0.F, 928.F, wf, 936.F},
                             {0.F, 944.F, wf, 952.F},
                             {0.F, 9.60e+2F, wf, 968.F},
                             {0.F, 976.F, wf, 984.F},
                             {0.F, 992.F, wf, 1.00e+3F},
                             {0.F, 1008.F, wf, 1016.F},
                             {0.F, 1024.F, wf, 1032.F},
                             {0.F, 1.040e+3F, wf, 1048.F},
                             {0.F, 1056.F, wf, 1064.F},
                             {0.F, 1072.F, wf, 1.080e+3F},
                             {0.F, 1088.F, wf, 1096.F},
                             {0.F, 1104.F, wf, 1112.F},
                             {0.F, 1.120e+3F, wf, 1128.F},
                             {0.F, 1136.F, wf, 1144.F},
                             {0.F, 1152.F, wf, 1.160e+3F},
                             {0.F, 1168.F, wf, 1176.F},
                             {0.F, 1184.F, wf, 1192.F},
                             {0.F, 1.20e+3F, wf, 1208.F},
                             {0.F, 1216.F, wf, 1224.F},
                             {0.F, 1232.F, wf, 1.240e+3F},
                             {0.F, 1248.F, wf, 1256.F},
                             {0.F, 1264.F, wf, 1272.F},
                             {0.F, 1.280e+3F, wf, 1288.F},
                             {0.F, 1296.F, wf, 1304.F},
                             {0.F, 1312.F, wf, 1.320e+3F},
                             {0.F, 1328.F, wf, 1336.F},
                             {0.F, 1344.F, wf, 1352.F},
                             {0.F, 1.360e+3F, wf, 1368.F},
                             {0.F, 1376.F, wf, 1384.F},
                             {0.F, 1392.F, wf, 1.40e+3F},
                             {0.F, 1408.F, wf, 1416.F},
                             {0.F, 1424.F, wf, 1432.F},
                             {0.F, 1.440e+3F, wf, 1448.F},
                             {0.F, 1456.F, wf, 1464.F},
                             {0.F, 1472.F, wf, 1.480e+3F},
                             {0.F, 1488.F, wf, 1496.F},
                             {0.F, 1504.F, wf, 1512.F},
                             {0.F, 1.520e+3F, wf, 1528.F},
                             {0.F, 1536.F, wf, 1544.F},
                             {0.F, 1552.F, wf, 1.560e+3F},
                             {0.F, 1568.F, wf, 1576.F},
                             {0.F, 1584.F, wf, 1592.F},
                             {0.F, 1.60e+3F, wf, 1608.F},
                             {0.F, 1616.F, wf, 1624.F},
                             {0.F, 1632.F, wf, 1.640e+3F},
                             {0.F, 1648.F, wf, 1656.F},
                             {0.F, 1664.F, wf, 1672.F},
                             {0.F, 1.680e+3F, wf, 1688.F},
                             {0.F, 1696.F, wf, 1704.F},
                             {0.F, 1712.F, wf, 1.720e+3F},
                             {0.F, 1728.F, wf, 1736.F},
                             {0.F, 1744.F, wf, 1752.F},
                             {0.F, 1.760e+3F, wf, 1768.F},
                             {0.F, 1776.F, wf, 1784.F},
                             {0.F, 1792.F, wf, 1.80e+3F},
                             {0.F, 1808.F, wf, 1816.F},
                             {0.F, 1824.F, wf, 1832.F},
                             {0.F, 1.840e+3F, wf, 1848.F},
                             {0.F, 1856.F, wf, 1864.F},
                             {0.F, 1872.F, wf, 1.880e+3F},
                             {0.F, 1888.F, wf, 1896.F},
                             {0.F, 1904.F, wf, 1912.F},
                             {0.F, 1.920e+3F, wf, 1928.F},
                             {0.F, 1936.F, wf, 1944.F},
                             {0.F, 1952.F, wf, 1.960e+3F},
                             {0.F, 1968.F, wf, 1976.F},
                             {0.F, 1984.F, wf, 1992.F},
                             {0.F, 2.00e+3F, wf, 2008.F},
                             {0.F, 2016.F, wf, 2024.F},
                             {0.F, 2032.F, wf, 2.040e+3F}}}}}};
  auto GLFW{glfw::init()};
  auto hints{glfw::WindowHints{.clientApi = glfw::ClientApi::OpenGl,
                               .contextVersionMajor = 2,
                               .contextVersionMinor = 0}};
  hints.apply();
  // from left to right: stripe pattern, frameId barcode, uniform
  auto idStripeWidth{16};
  auto idBits{16};
  auto wId{idStripeWidth * idBits};
  auto wUniform{128};
  auto wAll{w + wId + wUniform};
  auto window{glfw::Window(wAll, h, "GLFWPP Grating")};
  glfw::makeContextCurrent(window);
  // an alternative to increase swap interval is to change screen update rate
  // `xrandr --output HDMI-A-0 --mode 1920x1080 --rate 24`
  glfw::swapInterval(swapInterval);
  auto frameIdTimes2{0};
  while (!window.shouldClose()) {
    auto time{glfw::getTime()};
    glfw::pollEvents();
    if (frameIdTimes2 < 2 * drawFrames.size()) {
      frameIdTimes2++;
    } else {
      frameIdTimes2 = 0;
    }
    auto frameId{frameIdTimes2 / 2};
    glPushMatrix();
    // scale coordinates so that 0..w-1, 0..h-1 cover the screen
    glTranslatef(-1.0F, -1.0F, 0.F);
    glScalef(2.0F / wAll, 2.0F / h, 1.0F);
    drawFrames[frameId].execute();
    const auto hamming3{
        array<int, 16>({0, 3, 7, 4, 6, 5, 1, 2, 5, 6, 2, 1, 3, 0, 4, 7})};
    // frameId needs more than 4 bits. split into two 4 bit nibbles and encode
    // with (7,4) binary block code that can correct one bit error
    auto frameIdLo{(frameId & 0xF)};
    auto frameIdHi{((frameId & 0xF0)) >> 4};
    auto codedFrameIdLo{frameIdLo + (hamming3.at(frameIdLo) << 4)};
    auto codedFrameIdHi{frameIdHi + (hamming3.at(frameIdHi) << 4)};
    auto boundedCodedFrameId{1 + (codedFrameIdLo << 1) + (codedFrameIdHi << 8) +
                             (1 << 15)};
    // the codedFrameId consists of |S|Data|E|, with S and E being start and end
    // bits, respectively parity check via (7,4) binary block code (numerical
    // recipes 16.2.1 error correction codes and soft decision
    drawBarcode(boundedCodedFrameId, 16, 16, w, w + wId, 0, h);
    // uniform area that flickers between bright and dark in every frame
    if (0 == (frameId % 2)) {
      glColor4f(1.0F, 1.0F, 1.0F, 1.0F);
    } else {
      glColor4f(0.F, 0.F, 0.F, 1.0F);
    }
    glBegin(GL_QUADS);
    auto x0{w + wId};
    auto x1{wAll};
    auto y0{0};
    auto y1{h};
    glVertex2f(x0, y0);
    glVertex2f(x1, y0);
    glVertex2f(x1, y1);
    glVertex2f(x0, y1);
    glEnd();
    glPopMatrix();
    window.swapBuffers();
    frameDelayEstimator.update();
  }
  return 0;
}
