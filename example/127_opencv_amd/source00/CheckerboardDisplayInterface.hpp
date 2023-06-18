#pragma once
#include <opencv2/aruco.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>
class CheckerboardDisplayInterface {
public:
  virtual ~CheckerboardDisplayInterface() = default;
  virtual void
  displayCheckerboard(int squaresX, int squaresY, int squareLength,
                      cv::Ptr<cv::aruco::Dictionary> dictionary) = 0;
};
