// no preamble
#include "ArucoCheckerboardDisplay.h"
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>
void ArucoCheckerboardDisplay::displayCheckerboard(
    int squaresX, int squaresY, int squareLength,
    cv::Ptr<cv::aruco::Dictionary> dictionary) {
 /* auto board = cv::makePtr<cv::aruco::CharucoBoard>(
      cv::Size(squaresX, squaresY), 1.0f * squareLength, 0.50f * squareLength,
      dictionary, cv::noArray());*/
cv::Ptr<cv::aruco::CharucoBoard> boardPtr = cv::makePtr<cv::aruco::CharucoBoard>(
    cv::Size(squaresX, squaresY),
    1.0f * squareLength,
    0.50f * squareLength,
    dictionary
);

  board->draw(cv::Size(800, 600), board_image_, 10, 1);
  cv::imshow("checkerboard", board_image_);
  cv::waitKey(0);
}
