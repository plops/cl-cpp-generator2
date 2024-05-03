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
  auto w{1920 / 2};
  auto h{1080 / 2};
  cv::namedWindow(win, cv::WINDOW_NORMAL);
  cv::moveWindow(win, w, 100);
  cv::resizeWindow(win, w, h);
  auto x{20};
  auto y{270};
  auto clipLimit{13};
  auto screen{Screenshot(static_cast<int>(x), static_cast<int>(y), w, h)};
  cv::createTrackbar(
      "x", "x", &x, 1920 - w,
      [](int value, void *v) {
        auto screen{reinterpret_cast<Screenshot *>(v)};
        screen->SetX(value);
      },
      reinterpret_cast<void *>(&screen));
  cv::createTrackbar(
      "y", "y", &y, 1080 - h,
      [](int value, void *v) {
        auto screen{reinterpret_cast<Screenshot *>(v)};
        screen->SetY(value);
      },
      reinterpret_cast<void *>(&screen));
  cv::createTrackbar("clipLimit", "clipLimit", &clipLimit, 100);
  try {
    while (true) {
      screen(img);
      auto lab{cv::Mat()};
      cv::cvtColor(img, lab, cv::COLOR_BGR2Lab);
      auto labChannels{std::vector<cv::Mat>()};
      cv::split(lab, labChannels);
      cv::Ptr<cv::CLAHE> clahe{cv::createCLAHE()};
      clahe->setClipLimit(clipLimit);
      auto claheImage{cv::Mat()};
      clahe->apply(labChannels[0], claheImage);
      claheImage.copyTo(labChannels[0]);
      auto processedImage{cv::Mat()};
      cv::merge(labChannels, lab);
      cv::cvtColor(lab, processedImage, cv::COLOR_Lab2BGR);
      cv::imshow(win, processedImage);
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
