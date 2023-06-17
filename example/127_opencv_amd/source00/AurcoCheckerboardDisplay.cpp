// no preamble

#include "AurcoCheckerboardDisplay.h"
void AurcoCheckerboardDisplay::displayCheckerboard(
    int squaresX, int squaresY, int squareLength,
    cv::Ptr<cv::aruco::Dictionary> dictionary) {
  cv::aruco::drawCharucoBoard(boardSize, squareLength, squareLength / 2,
                              dictionary, boardImage);
  cv::imshow("checkerboard", boardImage);
  cv::waitKey(0);
}
