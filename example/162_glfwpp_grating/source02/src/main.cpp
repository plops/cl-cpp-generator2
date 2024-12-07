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
  vector<DrawFrame> drawFrames = {{.id = 0,
                                   .name = "bright",
                                   .draw = {{.color = {0.10F, 0.20F, 0.30F},
                                             .type = GL_QUADS,
                                             .coords = {{0, 0, 512, 512}}}}}};
  auto GLFW{glfw::init()};
  auto hints{glfw::WindowHints{.clientApi = glfw::ClientApi::OpenGl,
                               .contextVersionMajor = 2,
                               .contextVersionMinor = 0}};
  hints.apply();
  auto idStripeWidth{16};
  auto idBits{9};
  auto wId{idStripeWidth * idBits};
  auto w{512};
  auto wAll{w + wId};
  auto h{512};
  auto window{glfw::Window(wAll, h, "GLFWPP Grating")};
  glfw::makeContextCurrent(window);
  // an alternative to increase swap interval is to change screen update rate
  // `xrandr --output HDMI-A-0 --mode 1920x1080 --rate 24`
  glfw::swapInterval(swapInterval);
  while (!window.shouldClose()) {
    auto time{glfw::getTime()};
    glfw::pollEvents();
    // show a sequence of horizontal bars and vertical bars that split the image
    // into 1/2, 1/4th, ... . each image is followed by its inverted version.
    // the lcd of the projector is too slow to show this pattern exactly with
    // 60Hz. that is why we set swap interval to 2 (we wait for two frames for
    // every image so that the display has time to settle)
    static int current_level = 0;
    static bool horizontal = true;
    current_level = current_level + 1;
    if (current_level == 8 * 2) {
      horizontal = !horizontal;
      current_level = 0;
    }
    auto white{0 == (current_level % 2)};
    if (white) {
      glClearColor(dark, dark, dark, 1.0F);
      glClear(GL_COLOR_BUFFER_BIT);
      glColor4f(bright, bright, bright, 1.0F);
    } else {
      glClearColor(bright, bright, bright, 1.0F);
      glClear(GL_COLOR_BUFFER_BIT);
      glColor4f(dark, dark, dark, 1.0F);
    }
    glPushMatrix();
    // scale coordinates so that 0..w-1, 0..h-1 cover the screen
    glTranslatef(-1.0F, -1.0F, 0.F);
    glScalef(2.0F / wAll, 2.0F / h, 1.0F);
    glBegin(GL_QUADS);
    auto level{current_level / 2};
    auto y{1024 / pow(2.0F, level)};
    if (horizontal) {
      for (decltype(0 + pow(2.0F, level) + 1) i = 0; i < pow(2.0F, level);
           i += 1) {
        auto x{512};
        auto o{2 * i * y};
        glVertex2f(0, o);
        glVertex2f(0, o + y);
        glVertex2f(x, o + y);
        glVertex2f(x, o);
      }
    } else {
      for (decltype(0 + pow(2.0F, level) + 1) i = 0; i < pow(2.0F, level);
           i += 1) {
        auto x{512};
        auto o{2 * i * y};
        glVertex2f(o, 0);
        glVertex2f(o + y, 0);
        glVertex2f(o + y, x);
        glVertex2f(o, x);
      }
    }
    glEnd();
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
