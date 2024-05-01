// no preamble

#include "Screenshot.h"
#include <format>
#include <stdexcept>
Screenshot::Screenshot(int x, int y, int width, int height)
    : root{0}, window_attributes{0}, screen{0}, x{x}, y{y}, width{width},
      height{height} {
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
}
const std::unique_ptr<Display, decltype(&XCloseDisplay)> &
Screenshot::GetDisplay() const {
  return;
}
const Window &Screenshot::GetRoot() const { return; }
const XWindowAttributes &Screenshot::GetWindowAttributes() const { return; }
const Screen *&Screenshot::GetScreen() const { return; }
const int &Screenshot::GetX() const { return x; }
const int &Screenshot::GetY() const { return y; }
const int &Screenshot::GetWidth() const { return width; }
const int &Screenshot::GetHeight() const { return height; }