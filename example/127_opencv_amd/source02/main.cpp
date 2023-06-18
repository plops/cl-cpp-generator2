#include <iostream>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  auto x = 8;
  auto y = 3;
  auto square_len = 4.00e-2f;
  auto dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

  return 0;
}
