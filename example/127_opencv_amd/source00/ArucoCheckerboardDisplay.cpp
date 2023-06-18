// no preamble
#include "ArucoCheckerboardDisplay.h"
#include <opencv2/aruco.hpp>
#include <opencv2/charuco.hpp>
#include <opencv2/highgui.hpp>
void ArucoCheckerboardDisplay::displayCheckerboard(
    int squaresX, int squaresY, int squareLength,
    cv::Ptr<cv::aruco::Dictionary> dictionary) {
  auto board = cv::aruco::CharucoBoard::create(squaresX, squaresY, squareLength,
                                               squareLength / 2, dictionary);

  board->draw(cv::Size(800, 600), board_image_, 10, 1);
  cv::imshow("checkerboard", board_image_);
  cv::waitKey(0);
}
