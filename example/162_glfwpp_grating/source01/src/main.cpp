#include "/home/martin/src/popl/include/popl.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <format>
#include <glfwpp/glfwpp.h>
#include <iostream>
#include <numeric>
#include <thread>
#include <valarray>
using namespace std;
using namespace chrono;
using Scalar = float;
using Vec = std::vector<Scalar>;
using VecI = const Vec;

int main(int argc, char **argv) {
  auto op{popl::OptionParser("allowed options")};
  auto swapInterval{int(2)};
  auto helpOption{op.add<popl::Switch>("h", "help", "produce help message")};
  auto verboseOption{
      op.add<popl::Switch>("v", "verbose", "produce verbose output")};
  auto swapIntervalOption{op.add<popl::Value<int>>(
      "s", "swapInterval", "parameter", 2, &swapInterval)};
  op.parse(argc, argv);
  if (helpOption->is_set()) {
    cout << op << endl;
    exit(0);
  }
  auto computeStat{[](const auto &fitres,
                      auto filter) -> tuple<Scalar, Scalar, Scalar, Scalar> {
    // compute mean and standard deviation Numerical Recipes 14.1.2 and 14.1.8
    auto data{valarray<Scalar>(fitres.size())};
    data.resize(fitres.size());
    transform(fitres.begin(), fitres.end(), &data[0], filter);
    const auto N{static_cast<Scalar>(data.size())};
    const auto mean{(data.sum()) / N};
    const auto stdev{
        sqrt(((pow(data - mean, 2).sum()) - (pow((data - mean).sum(), 2) / N)) /
             (N - 1.0F))};
    const auto mean_stdev{stdev / sqrt(N)};
    const auto stdev_stdev{stdev / sqrt(2 * N)};
    return make_tuple(mean, mean_stdev, stdev, stdev_stdev);
  }};
  auto fitres{deque<float>()};
  auto GLFW{glfw::init()};
  auto hints{glfw::WindowHints{.clientApi = glfw::ClientApi::OpenGl,
                               .contextVersionMajor = 2,
                               .contextVersionMinor = 0}};
  hints.apply();
  auto w{512};
  auto h{512};
  auto window{glfw::Window(w, h, "GLFWPP Grating")};
  glfw::makeContextCurrent(window);
  // an alternative to increase swap interval is to change screen update rate
  // `xrandr --output HDMI-A-0 --mode 1920x1080 --rate 24`
  glfw::swapInterval(swapInterval);
  auto t0{high_resolution_clock::now()};
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
      glClearColor(0.F, 0.F, 0.F, 1.0F);
      glClear(GL_COLOR_BUFFER_BIT);
      glColor4f(1.0F, 1.0F, 1.0F, 1.0F);
    } else {
      glClearColor(1.0F, 1.0F, 1.0F, 1.0F);
      glClear(GL_COLOR_BUFFER_BIT);
      glColor4f(0.F, 0.F, 0.F, 1.0F);
    }
    glPushMatrix();
    glTranslatef(-1.0F, -1.0F, 0.F);
    glScalef(2.0F / w, 2.0F / h, 1.0F);
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
    glPopMatrix();
    window.swapBuffers();
    auto t1{high_resolution_clock::now()};
    auto frameTimens{duration_cast<nanoseconds>(t1 - t0).count()};
    auto frameTimems{frameTimens / 1.0e+6F};
    auto frameRateHz{1.0e+9F / frameTimens};
    fitres.push_back(frameTimems);
    if (67 < fitres.size()) {
      fitres.pop_front();
    }
    auto [frameTime_, frameTime_Std, frameTimeStd, frameTimeStdStd]{
        computeStat(fitres, [&](const auto &f) { return f; })};
    std::cout << std::format("(:frameTime_ '{}' :frameTimeStd '{}' "
                             ":frameTimems '{}' :frameRateHz '{}')\n",
                             frameTime_, frameTimeStd, frameTimems,
                             frameRateHz);
    t0 = t1;
  }
  return 0;
}
