#include <iostream>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  auto camera = VideoCapture(0);
  auto title = "Webcam";
  auto x = 8;
  auto y = 3;
  auto square_len = 4.00e-2f;
  auto dict0 = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
  auto dict = makePtr<aruco::Dictionary>(
      aruco::getPredefinedDictionary(aruco::DICT_6X6_250));
  auto board = new aruco::CharucoBoard(Size(x, y), square_len,
                                       0.50f * square_len, dict0);
  auto img = Mat();
  board->generateImage(cv::Size(800, 600), img, 10, 1);

  if (!camera.isOpened()) {
    std::cout << "Error: Could not open camera." << std::endl;
    return 1;
  }
  namedWindow(title, WINDOW_AUTOSIZE);
  auto waitTime = 10;
  // waitTime in milliseconds

  auto frame = Mat();
  auto ids = std::vector<int>();
  auto corners = std::vector<std::vector<Point2f>>();
  auto allCorners = std::vector<std::vector<std::vector<Point2f>>>();
  auto allIds = std::vector<std::vector<int>>();
  while (true) {
    // capture image

    camera >> frame;
    if (frame.empty()) {
      break;
    }

    // detect markers

    aruco::detectMarkers(frame, dict, corners, ids);

    if (0 < ids.size()) {
      // interpolate charuco corners

      auto charucoCorners = Mat();
      auto charucoIds = Mat();
      aruco::interpolateCornersCharuco(corners, ids, frame, board,
                                       charucoCorners, charucoIds);
      if (0 < charucoCorners.total()) {
        // If at leas one charuco corner detected, draw the corners

        aruco::drawDetectedCornersCharuco(frame, charucoCorners, charucoIds);
        // Collect data for calibration

        allCorners.push_back(corners);
        allIds.push_back(ids);
      }
    }
    imshow(title, frame);
    if (0 <= waitKey(1)) {
      break;
    }
  }

  return 0;
}
