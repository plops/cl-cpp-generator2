#include <iostream>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  auto x = 8;
  auto y = 3;
  auto square_len = 4.00e-2f;
  auto dict = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
  Ptr<aruco::CharucoBoard> board =
      new aruco::CharucoBoard(Size(x, y), square_len, 0.50f * square_len, dict);

  return 0;
}
