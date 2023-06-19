#include <iostream>
#include <opencv2/aruco.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  auto camera = VideoCapture(0);
  auto title = "Webcam";
  if (!camera.isOpened()) {
    std::cout << "Error: Could not open camera." << std::endl;
    return 1;
  }
  namedWindow(title, WINDOW_AUTOSIZE);
  auto dict = makePtr<aruco::Dictionary>(
      aruco::getPredefinedDictionary(aruco::DICT_6X6_250));
  auto frame = Mat();
  auto markerIds = std::vector<int>();
  auto markerCorners = std::vector<std::vector<Point2f>>();
  while (true) {
    camera >> frame;
    if (frame.empty()) {
      break;
    }
    aruco::detectMarkers(frame, dict, corners, ids);
    if (0 < ids.size()) {
      aruco::drawDetectedMarkers(frame, corners, ids);
    }
    imshow(title, frame);
    if (0 <= waitKey(1)) {
      break;
    }
  }

  return 0;
}
