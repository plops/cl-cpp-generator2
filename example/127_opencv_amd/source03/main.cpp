#include <iostream>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  auto x = 8;
  auto y = 3;
  auto square_len = 4.00e-2f;
  auto dict = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
  auto board =
      new aruco::CharucoBoard(Size(x, y), square_len, 0.50f * square_len, dict);
  auto img = Mat();
  board->generateImage(cv::Size(800, 600), img, 10, 1);
  cv::imshow("charuco board", board);
  cv::waitKey(0);

  return 0;
}
