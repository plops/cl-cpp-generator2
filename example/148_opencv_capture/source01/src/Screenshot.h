#ifndef SCREENSHOT_H
#define SCREENSHOT_H

#include <opencv2/opencv.hpp>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XShm.h>
#include <sys/ipc.h>
#include <sys/shm.h> 
class Screenshot  {
        public:
        explicit  Screenshot (int x, int y, int width, int height)       ;   
        void operator() (cv::Mat& cv_img)       ;   
        std::unique_ptr<Display, decltype(&XCloseDisplay)> display;
        Window root;
        XWindowAttributes window_attributes;
        Screen* screen;
        XShmSegmentInfo shminfo;
        std::unique_ptr<XImage, void (*) (XImage*)> ximg;
        int x;
        int y;
        int width;
        int height;
};

#endif /* !SCREENSHOT_H */