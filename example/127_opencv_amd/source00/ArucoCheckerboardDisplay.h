#ifndef ARUCOCHECKERBOARDDISPLAY_H
#define ARUCOCHECKERBOARDDISPLAY_H

#include "CheckerboardDisplayInterface.hpp"

class ArucoCheckerboardDisplay : public CheckerboardDisplayInterface {
        public:
        void displayCheckerboard (int squaresX, int squaresY, int squareLength, cv::Ptr<cv::aruco::Dictionary> dictionary)       ;  
        private:
        int squares_x_;
        int squares_y_;
        int square_length_;
        cv::Ptr<cv::aruco::Dictionary> dictionary_;
        cv::Mat board_image_;
};

#endif /* !ARUCOCHECKERBOARDDISPLAY_H */