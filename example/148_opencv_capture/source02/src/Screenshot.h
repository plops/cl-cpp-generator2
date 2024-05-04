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
        Display* GetDisplay ()       ;   
        void SetDisplay (Display* display)       ;   
        const bool& GetInit () const      ;   
        void SetInit (bool init)       ;   
        const Window& GetRoot () const      ;   
        void SetRoot (Window root)       ;   
        const XWindowAttributes& GetWindowAttributes () const      ;   
        void SetWindowAttributes (XWindowAttributes window_attributes)       ;   
        Screen* GetScreen ()       ;   
        void SetScreen (Screen* screen)       ;   
        const XShmSegmentInfo& GetShminfo () const      ;   
        void SetShminfo (XShmSegmentInfo shminfo)       ;   
        XImage* GetXimg ()       ;   
        void SetXimg (XImage* ximg)       ;   
        const int& GetX () const      ;   
        void SetX (int x)       ;   
        const int& GetY () const      ;   
        void SetY (int y)       ;   
        const int& GetWidth () const      ;   
        void SetWidth (int width)       ;   
        const int& GetHeight () const      ;   
        void SetHeight (int height)       ;   
        private:
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