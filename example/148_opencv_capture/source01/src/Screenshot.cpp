// no preamble

#include "Screenshot.h"
#include <format>
#include <memory>
#include <stdexcept>
Screenshot::Screenshot(int x_, int y_, int width_, int height_)
    : x{x_}, y{y_}, width{width_}, height{height_} {
  auto d{XOpenDisplay(nullptr)};
  if (!d) {
    throw std::runtime_error("Failed to open display");
  }
  display = d;
  root = DefaultRootWindow(*display);
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