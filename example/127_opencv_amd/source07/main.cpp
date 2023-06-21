#include <iostream>
#include <opencv2/objdetect/aruco_board.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

void print(aruco::DetectorParameters const &p) {
  std::cout << "DetectorParameters"
            << " p.adaptiveThreshWinSizeMin='" << p.adaptiveThreshWinSizeMin
            << "' " << std::endl;
  std::cout << "DetectorParameters"
            << " p.adaptiveThreshWinSizeMax='" << p.adaptiveThreshWinSizeMax
            << "' " << std::endl;
  std::cout << "DetectorParameters"
            << " p.adaptiveThreshWinSizeStep='" << p.adaptiveThreshWinSizeStep
            << "' " << std::endl;
  std::cout << "DetectorParameters"
            << " p.adaptiveThreshConstant='" << p.adaptiveThreshConstant << "' "
            << std::endl;
  std::cout << "DetectorParameters"
            << " p.minMarkerPerimeterRate='" << p.minMarkerPerimeterRate << "' "
            << std::endl;
  std::cout << "DetectorParameters"
            << " p.maxMarkerPerimeterRate='" << p.maxMarkerPerimeterRate << "' "
            << std::endl;
  std::cout << "DetectorParameters"
            << " p.polygonalApproxAccuracyRate='"
            << p.polygonalApproxAccuracyRate << "' " << std::endl;
  std::cout << "DetectorParameters"
            << " p.minCornerDistanceRate='" << p.minCornerDistanceRate << "' "
            << std::endl;
  std::cout << "DetectorParameters"
            << " p.minDistanceToBorder='" << p.minDistanceToBorder << "' "
            << std::endl;
  std::cout << "DetectorParameters"
            << " p.minMarkerDistanceRate='" << p.minMarkerDistanceRate << "' "
            << std::endl;
  std::cout << "DetectorParameters"
            << " p.cornerRefinementMethod='" << p.cornerRefinementMethod << "' "
            << std::endl;
  std::cout << "DetectorParameters"
            << " p.cornerRefinementWinSize='" << p.cornerRefinementWinSize
            << "' " << std::endl;
  std::cout << "DetectorParameters"
            << " p.cornerRefinementMaxIterations='"
            << p.cornerRefinementMaxIterations << "' " << std::endl;
  std::cout << "DetectorParameters"
            << " p.cornerRefinementMinAccuracy='"
            << p.cornerRefinementMinAccuracy << "' " << std::endl;
  std::cout << "DetectorParameters"
            << " p.markerBorderBits='" << p.markerBorderBits << "' "
            << std::endl;
  std::cout << "DetectorParameters"
            << " p.perspectiveRemovePixelPerCell='"
            << p.perspectiveRemovePixelPerCell << "' " << std::endl;
  std::cout << "DetectorParameters"
            << " p.perspectiveRemoveIgnoredMarginPerCell='"
            << p.perspectiveRemoveIgnoredMarginPerCell << "' " << std::endl;
  std::cout << "DetectorParameters"
            << " p.maxErroneousBitsInBorderRate='"
            << p.maxErroneousBitsInBorderRate << "' " << std::endl;
  std::cout << "DetectorParameters"
            << " p.minOtsuStdDev='" << p.minOtsuStdDev << "' " << std::endl;
  std::cout << "DetectorParameters"
            << " p.errorCorrectionRate='" << p.errorCorrectionRate << "' "
            << std::endl;
  std::cout << "DetectorParameters"
            << " p.aprilTagQuadDecimate='" << p.aprilTagQuadDecimate << "' "
            << std::endl;
  std::cout << "DetectorParameters"
            << " p.aprilTagQuadSigma='" << p.aprilTagQuadSigma << "' "
            << std::endl;
  std::cout << "DetectorParameters"
            << " p.aprilTagMinClusterPixels='" << p.aprilTagMinClusterPixels
            << "' " << std::endl;
  std::cout << "DetectorParameters"
            << " p.aprilTagMaxNmaxima='" << p.aprilTagMaxNmaxima << "' "
            << std::endl;
  std::cout << "DetectorParameters"
            << " p.aprilTagCriticalRad='" << p.aprilTagCriticalRad << "' "
            << std::endl;
  std::cout << "DetectorParameters"
            << " p.aprilTagMaxLineFitMse='" << p.aprilTagMaxLineFitMse << "' "
            << std::endl;
  std::cout << "DetectorParameters"
            << " p.aprilTagMinWhiteBlackDiff='" << p.aprilTagMinWhiteBlackDiff
            << "' " << std::endl;
  std::cout << "DetectorParameters"
            << " p.aprilTagDeglitch='" << p.aprilTagDeglitch << "' "
            << std::endl;
  std::cout << "DetectorParameters"
            << " p.detectInvertedMarker='" << p.detectInvertedMarker << "' "
            << std::endl;
  std::cout << "DetectorParameters"
            << " p.useAruco3Detection='" << p.useAruco3Detection << "' "
            << std::endl;
  std::cout << "DetectorParameters"
            << " p.minSideLengthCanonicalImg='" << p.minSideLengthCanonicalImg
            << "' " << std::endl;
  std::cout << "DetectorParameters"
            << " p.minMarkerLengthRatioOriginalImg='"
            << p.minMarkerLengthRatioOriginalImg << "' " << std::endl;
}

void print(aruco::RefineParameters const &p) {
  std::cout << "RefineParameters"
            << " p.minRepDistance='" << p.minRepDistance << "' " << std::endl;
  std::cout << "RefineParameters"
            << " p.errorCorrectionRate='" << p.errorCorrectionRate << "' "
            << std::endl;
  std::cout << "RefineParameters"
            << " p.checkAllOrders='" << p.checkAllOrders << "' " << std::endl;
}

void print(aruco::CharucoParameters const &p) {
  std::cout << "CharucoParameters"
            << " p.minMarkers='" << p.minMarkers << "' " << std::endl;
  std::cout << "CharucoParameters"
            << " p.tryRefineMarkers='" << p.tryRefineMarkers << "' "
            << std::endl;
}

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
  auto board = std::make_unique<aruco::CharucoBoard>(
      aruco::CharucoBoard(Size(x, y), square_len, 0.50f * square_len, *dict));
  auto img = Mat();
  board->generateImage(cv::Size(800, 600), img, 10, 1);
  auto dp = aruco::DetectorParameters();
  print(dp);

  auto rp = aruco::RefineParameters();
  print(rp);

  auto cp = aruco::CharucoParameters();
  print(cp);

  auto markerDetector = aruco::ArucoDetector(*dict, dp, rp);
  auto boardDetector = aruco::CharucoDetector(*board, cp, dp, rp);

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

    auto key = (char)waitKey(waitTime);
    camera >> frame;
    if (frame.empty() || (key == 27)) {
      break;
    }

    // detect markers

    markerDetector.detectMarkers(frame, corners, ids, marker_rejected);
    markerDetector.refineDetectedMarkers(frame, *board, corners, ids,
                                         marker_rejected);

    if (!ids.empty()) {
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
  }

  return 0;
}
