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
         ~Screenshot ()       ;   
        void operator() (cv::Mat& cv_img)       ;   
        Display* display;
        bool init;
        Window root;
        XWindowAttributes window_attributes;
        Screen* screen;
        XShmSegmentInfo shminfo;
        XImage* ximg;
        int x;
        int y;
        int width;
        int height;
};

#endif /* !SCREENSHOT_H */