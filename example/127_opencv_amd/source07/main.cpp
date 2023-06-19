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
  auto dict = makePtr<aruco::Dictionary>(
      aruco::getPredefinedDictionary(aruco::DICT_6X6_250));
  auto board = new aruco::CharucoBoard(Size(x, y), square_len,
                                       0.50f * square_len, *dict);
  auto img = Mat();
  board->generateImage(cv::Size(800, 600), img, 10, 1);
  auto markerDetector = aruco::ArucoDetector(*dict);
  auto boardDetector = aruco::CharucoDetector(*board);

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
  auto marker_rejected = std::vector<std::vector<Point2f>>();
  auto allCorners = std::vector<Mat>();
  auto allIds = std::vector<Mat>();
  while (true) {
    // capture image

    camera >> frame;
    if (frame.empty()) {
      break;
    }

    // detect markers

    markerDetector.detectMarkers(frame, corners, ids, marker_rejected);
    markerDetector.refineDetectedMarkers(frame, *board, corners, ids,
                                         marker_rejected);

    if (0 < ids.size()) {
      std::cout << ""
                << " ids.size()='" << ids.size() << "' " << std::endl;
      // interpolate charuco corners (checker board corners, not the aruco
      // markers)

      auto charucoCorners = Mat();
      auto charucoIds = Mat();
      boardDetector.detectBoard(frame, charucoCorners, charucoIds, corners,
                                ids);
      aruco::drawDetectedMarkers(frame, corners, ids);
      if (4 <= charucoCorners.size().height) {
        std::cout << ""
                  << " charucoCorners.size().height='"
                  << charucoCorners.size().height << "' " << std::endl;
        aruco::drawDetectedCornersCharuco(frame, charucoCorners, charucoIds);
        allCorners.push_back(charucoCorners);
        allIds.push_back(charucoIds);
      }
    }
    imshow(title, frame);
    auto key = (char)waitKey(waitTime);
    if (key == 27) {
      break;
    }
  }
  if (0 < allIds.size()) {
    auto cameraMatrix = Mat();
    auto distCoeffs = Mat();
    auto rvecs = std::vector<Mat>();
    auto tvecs = std::vector<Mat>();
    auto repError =
        aruco::calibrateCameraCharuco(allCorners, allIds, board, Size(640, 480),
                                      cameraMatrix, distCoeffs, rvecs, tvecs);
    auto fs = FileStorage("calibration.yaml", FileStorage::WRITE);
    // rvecs and tvecs are camera pose estimates

    fs << "cameraMatrix" << cameraMatrix << "distCoeffs" << distCoeffs;
    fs.release();
  }

  return 0;
}
