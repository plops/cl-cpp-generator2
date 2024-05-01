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
  auto count{0};
  while (true) {
    count++;
  }
  return 0;
}
