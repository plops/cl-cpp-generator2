#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XShm.h>
#include <format>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <sys/ipc.h>
#include <sys/shm.h>

int main(int argc, char **argv) {
  std::cout << std::format("start\n");
  auto img{cv::Mat()};
  try {
    auto screen{Screenshot(0, 0, 1920, 1080)};
    screen(img);
    cv::imshow("img", img);
    cv::waitKey(0);
  } catch (const std::exception &e) {
    std::cout << std::format(" e.what()='{}'\n", e.what());
    return 1;
  }
  return 0;
}
