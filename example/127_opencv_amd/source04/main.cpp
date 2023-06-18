#include <iostream>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  auto dict = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
  auto img = imread("/home/martin/charucoboard.png", IMREAD_COLOR);
  auto markerIds = std::vector<int>();
  auto markerCorners = std::vector<std::vector<Point2f>>();
  aruco::detectMarkers(img, dict, markerCorners, markerIds);
  imshow("charuco board", img);
  waitKey(0);

  return 0;
}
