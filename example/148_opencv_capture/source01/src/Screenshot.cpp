// no preamble

#include "Screenshot.h"
#include <cerrno>
#include <cstring>
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
    std::cout << std::format(" errno='{}' strerror(errno)='{}'\n", errno,
                             strerror(errno));
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
  XShmGetImage(display, root, ximg, x, y, 0x00ffffff);
  cv_img = cv::Mat(height, width, CV_8UC4, ximg->data);
}
Display *Screenshot::GetDisplay() { return display; }
void Screenshot::SetDisplay(Display *display) { this->display = display; }
const bool &Screenshot::GetInit() const { return init; }
void Screenshot::SetInit(bool init) { this->init = init; }
const Window &Screenshot::GetRoot() const { return root; }
void Screenshot::SetRoot(Window root) { this->root = root; }
const XWindowAttributes &Screenshot::GetWindowAttributes() const {
  return window_attributes;
}
void Screenshot::SetWindowAttributes(XWindowAttributes window_attributes) {
  this->window_attributes = window_attributes;
}
Screen *Screenshot::GetScreen() { return screen; }
void Screenshot::SetScreen(Screen *screen) { this->screen = screen; }
const XShmSegmentInfo &Screenshot::GetShminfo() const { return shminfo; }
void Screenshot::SetShminfo(XShmSegmentInfo shminfo) {
  this->shminfo = shminfo;
}
XImage *Screenshot::GetXimg() { return ximg; }
void Screenshot::SetXimg(XImage *ximg) { this->ximg = ximg; }
const int &Screenshot::GetX() const { return x; }
void Screenshot::SetX(int x) { this->x = x; }
const int &Screenshot::GetY() const { return y; }
void Screenshot::SetY(int y) { this->y = y; }
const int &Screenshot::GetWidth() const { return width; }
void Screenshot::SetWidth(int width) { this->width = width; }
const int &Screenshot::GetHeight() const { return height; }
void Screenshot::SetHeight(int height) { this->height = height; }