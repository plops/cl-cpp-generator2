// no preamble

#include "Screenshot.h"
#include <format>
#include <memory>
#include <stdexcept>
Screenshot::Screenshot(int x, int y, int width, int height)
    : x{x}, y{y}, width{width}, height{height} {
  display = XOpenDisplay(nullptr);
  if (!display) {
    throw std::runtime_error("Failed to open display");
  }
  root = DefaultRootWindow(display);
  XGetWindowAttributes(display, root, &window_attributes);
  screen = window_attributes.screen;
  ximg = XShmCreateImage(display, DefaultVisualOfScreen(screen),
                         DefaultDepthOfScreen(screen), ZPixmap, nullptr,
                         &shminfo, width, height);
  shminfo.shmid = shmget(IPC_PRIVATE, ximg->bytes_per_line * ximg->height,
                         IPC_CREAT | 0777);
  if (shminfo.shmid < 0) {
    throw std::runtime_error("Fatal shminfo error!");
  }
  ximg->data = reinterpret_cast<char *>(shmat(shminfo.shmid, 0, 0));
  shminfo.shmaddr = ximg->data;
  shminfo.readOnly = False;
  if (!XShmAttach(display, &shminfo)) {
    throw std::runtime_error("XShmAttach failed");
  }
  init = true;
}
Screenshot::~Screenshot() {
  if (!init) {
    XDestroyImage(ximg);
  }
  XShmDetach(display, &shminfo);
  shmdt(shminfo.shmaddr);
  XCloseDisplay(display);
}
void Screenshot::operator()(cv::Mat &cv_img) {
  if (init) {
    init = false;
  }
  XShmGetImage(display, root, ximg, 0, 0, 0x00ffffff);
  cv_img = cv::Mat(height, width, CV_8UC4, ximg->data);
}
Display *Screenshot::GetDisplay() { return display; }
const bool &Screenshot::GetInit() const { return init; }
const Window &Screenshot::GetRoot() const { return root; }
const XWindowAttributes &Screenshot::GetWindowAttributes() const {
  return window_attributes;
}
Screen *Screenshot::GetScreen() { return screen; }
const XShmSegmentInfo &Screenshot::GetShminfo() const { return shminfo; }
XImage *Screenshot::GetXimg() { return ximg; }
const int &Screenshot::GetX() const { return x; }
const int &Screenshot::GetY() const { return y; }
const int &Screenshot::GetWidth() const { return width; }
const int &Screenshot::GetHeight() const { return height; }