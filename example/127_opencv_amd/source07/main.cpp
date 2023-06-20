#include <iostream>
#include <opencv2/objdetect/aruco_board.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
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
  auto dp = aruco::DetectorParameters();
  std::cout << ""
            << " dp.adaptiveThreshWinSizeMin='" << dp.adaptiveThreshWinSizeMin
            << "' " << std::endl;
  std::cout << ""
            << " dp.adaptiveThreshWinSizeMax='" << dp.adaptiveThreshWinSizeMax
            << "' " << std::endl;
  std::cout << ""
            << " dp.adaptiveThreshWinSizeStep='" << dp.adaptiveThreshWinSizeStep
            << "' " << std::endl;
  std::cout << ""
            << " dp.adaptiveThreshConstant='" << dp.adaptiveThreshConstant
            << "' " << std::endl;
  std::cout << ""
            << " dp.minMarkerPerimeterRate='" << dp.minMarkerPerimeterRate
            << "' " << std::endl;
  std::cout << ""
            << " dp.maxMarkerPerimeterRate='" << dp.maxMarkerPerimeterRate
            << "' " << std::endl;
  std::cout << ""
            << " dp.polygonalApproxAccuracyRate='"
            << dp.polygonalApproxAccuracyRate << "' " << std::endl;
  std::cout << ""
            << " dp.minCornerDistanceRate='" << dp.minCornerDistanceRate << "' "
            << std::endl;
  std::cout << ""
            << " dp.minDistanceToBorder='" << dp.minDistanceToBorder << "' "
            << std::endl;
  std::cout << ""
            << " dp.minMarkerDistanceRate='" << dp.minMarkerDistanceRate << "' "
            << std::endl;
  std::cout << ""
            << " dp.cornerRefinementMethod='" << dp.cornerRefinementMethod
            << "' " << std::endl;
  std::cout << ""
            << " dp.cornerRefinementWinSize='" << dp.cornerRefinementWinSize
            << "' " << std::endl;
  std::cout << ""
            << " dp.cornerRefinementMaxIterations='"
            << dp.cornerRefinementMaxIterations << "' " << std::endl;
  std::cout << ""
            << " dp.cornerRefinementMinAccuracy='"
            << dp.cornerRefinementMinAccuracy << "' " << std::endl;
  std::cout << ""
            << " dp.markerBorderBits='" << dp.markerBorderBits << "' "
            << std::endl;
  std::cout << ""
            << " dp.perspectiveRemovePixelPerCell='"
            << dp.perspectiveRemovePixelPerCell << "' " << std::endl;
  std::cout << ""
            << " dp.perspectiveRemoveIgnoredMarginPerCell='"
            << dp.perspectiveRemoveIgnoredMarginPerCell << "' " << std::endl;
  std::cout << ""
            << " dp.maxErroneousBitsInBorderRate='"
            << dp.maxErroneousBitsInBorderRate << "' " << std::endl;
  std::cout << ""
            << " dp.minOtsuStdDev='" << dp.minOtsuStdDev << "' " << std::endl;
  std::cout << ""
            << " dp.errorCorrectionRate='" << dp.errorCorrectionRate << "' "
            << std::endl;
  std::cout << ""
            << " dp.aprilTagQuadDecimate='" << dp.aprilTagQuadDecimate << "' "
            << std::endl;
  std::cout << ""
            << " dp.aprilTagQuadSigma='" << dp.aprilTagQuadSigma << "' "
            << std::endl;
  std::cout << ""
            << " dp.aprilTagMinClusterPixels='" << dp.aprilTagMinClusterPixels
            << "' " << std::endl;
  std::cout << ""
            << " dp.aprilTagMaxNmaxima='" << dp.aprilTagMaxNmaxima << "' "
            << std::endl;
  std::cout << ""
            << " dp.aprilTagCriticalRad='" << dp.aprilTagCriticalRad << "' "
            << std::endl;
  std::cout << ""
            << " dp.aprilTagMaxLineFitMse='" << dp.aprilTagMaxLineFitMse << "' "
            << std::endl;
  std::cout << ""
            << " dp.aprilTagMinWhiteBlackDiff='" << dp.aprilTagMinWhiteBlackDiff
            << "' " << std::endl;
  std::cout << ""
            << " dp.aprilTagDeglitch='" << dp.aprilTagDeglitch << "' "
            << std::endl;
  std::cout << ""
            << " dp.detectInvertedMarker='" << dp.detectInvertedMarker << "' "
            << std::endl;
  std::cout << ""
            << " dp.useAruco3Detection='" << dp.useAruco3Detection << "' "
            << std::endl;
  std::cout << ""
            << " dp.minSideLengthCanonicalImg='" << dp.minSideLengthCanonicalImg
            << "' " << std::endl;
  std::cout << ""
            << " dp.minMarkerLengthRatioOriginalImg='"
            << dp.minMarkerLengthRatioOriginalImg << "' " << std::endl;

  auto rp = aruco::RefineParameters();
  std::cout << ""
            << " rp.minRepDistance='" << rp.minRepDistance << "' " << std::endl;
  std::cout << ""
            << " rp.errorCorrectionRate='" << rp.errorCorrectionRate << "' "
            << std::endl;
  std::cout << ""
            << " rp.checkAllOrders='" << rp.checkAllOrders << "' " << std::endl;

  auto markerDetector = aruco::ArucoDetector(*dict, dp, rp);
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

  return 0;
}
