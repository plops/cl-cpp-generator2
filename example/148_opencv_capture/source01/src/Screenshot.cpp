// no preamble

#include "Screenshot.h"
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XShm.h>
#include <format>
#include <iostream>
#include <memory>
#include <opencv4/opencv2/opencv.hpp>
#include <stdexcept>
#include <sys/ipc.h>
#include <sys/shm.h>
Screenshot::Screenshot(int x, int y, int width, int height)
    : root{0}, window_attributes{0}, screen{0}, shminfo{0}, ximg{0}, x{x}, y{y},
      width{width}, height{height} {
  display = std::unique_ptr<Display, decltype(&XCloseDisplay)>(
      XOpenDisplay(nullptr), XCloseDisplay);
  if (!display) {
    throw std::runtime_error("Failed to open display");
  }
  root = DefaultRootWIndow(*display);
  XGetWindowAttributes(*display, root, &window_attributes);
  screen = window_attributes.screen;
  ximg = std::unique_ptr<XImage, void (*)(XImage *)>(
      XShmCreateImage(*display, DefaultVisualOfScreen(screen),
                      DefaultDepthOfScreen(screen), ZPixmap, nullptr, &shminfo,
                      width, height),
      [](XImage *img) {
        XShmDetach(*display, &shminfo);
        shmdt(shminfo.shmaddr);
        XDestroyImage(img);
      });
  shminfo.shmid = shmget(IPC_PRIVATE, ximg->bytes_per_line * ximg->height,
                         IPC_CREAT | 0777);
  if (shminfo.shmid < 0) {
    throw std::runtime_error("Fatal shminfo error!");
  }
  ximg->data = reinterpret_cast<char *>(shmat(shminfo.shmid, 0, 0));
  shminfo.shmaddr = ximg->data;
  shminfo.readOnly = False;
  if (!XShmAttach(*display, &shminfo)) {
    throw std::runtime_error("XShmAttach failed");
  }
}
void Screenshot::operator()(cv::Mat &cv_img) {
  XShmGetImage(*display, root, xshm.get(), 0, 0, 0x00ffffff);
  cv_img = cv::Mat(height, width, CV_8UC4, ximg->data);
}
const std::unique_ptr<Display, decltype(&XCloseDisplay)> &
Screenshot::GetDisplay() const {
  return;
}
const Window &Screenshot::GetRoot() const { return; }
const XWindowAttributes &Screenshot::GetWindowAttributes() const { return; }
const Screen *&Screenshot::GetScreen() const { return; }
const XShmSegmentInfo &Screenshot::GetShminfo() const { return; }
const std::unique_ptr<XImage, void (*)(XImage *)> &Screenshot::GetXimg() const {
  return;
}
const int &Screenshot::GetX() const { return x; }
const int &Screenshot::GetY() const { return y; }
const int &Screenshot::GetWidth() const { return width; }
const int &Screenshot::GetHeight() const { return height; }