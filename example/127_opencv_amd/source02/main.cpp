#include <iostream>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  // https://github.com/opencv/opencv_contrib/blob/7a4c0dfa861bbd4e5df7081949f685696eb9a94f/modules/aruco/samples/detect_board_charuco.cpp#L148

  auto x = 8;
  auto y = 3;
  auto square_len = 4.00e-2f;
  auto dict = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
  auto board =
      new aruco::CharucoBoard(Size(x, y), square_len, 0.50f * square_len, dict);
  auto img = Mat();
  board->generateImage(cv::Size(800, 600), img, 10, 1);
  if (!cv::imwrite("/home/martin/charucoboard.png", img)) {
    std::cout << "Failed to save the image." << std::endl;
    return -1;
  }

  std::cout << "charucoboard has been saved." << std::endl;
  return 0;
}
