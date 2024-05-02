#include "Screenshot.h"
#include <format>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
  std::cout << std::format("start\n");
  auto img{cv::Mat()};
  auto win{"img"};
  auto frameRate{60.F};
  auto alpha{0.20F};
  auto w{640};
  auto h{480};
  cv::namedWindow(win, cv::WINDOW_GUI_EXPANDED);
  cv::moveWindow(win, w, 100);
  cv::resizeWindow(win, w, h);
  try {
    while (true) {
      auto screen{Screenshot(0, 0, w, h)};
      screen(img);
      cv::imshow(win, img);
      if (27 == cv::waitKey(1000 / 60)) {
        // Exit loop if ESC key is pressed

        break;
      }
    }
  } catch (const std::exception &e) {
    std::cout << std::format(" e.what()='{}'\n", e.what());
    return 1;
  }
  return 0;
}
