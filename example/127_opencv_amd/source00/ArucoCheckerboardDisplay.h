#ifndef AURCOCHECKERBOARDDISPLAY_H
#define AURCOCHECKERBOARDDISPLAY_H

#include "CheckerboardDisplayInterface.hpp"

class AurcoCheckerboardDisplay  {
        public:
        void displayCheckerboard (int squaresX, int squaresY, int squareLength, cv::Ptr<cv::aruco::Dictionary> dictionary)       ;  
        private:
        int squares_x_;
        int squares_y_;
        int square_length_;
        cv::Ptr<cv::aruco::Dictionary> dictionary_;
        cv::Size board_size_;
        cv::Mat board_image_;
};

#endif /* !AURCOCHECKERBOARDDISPLAY_H */