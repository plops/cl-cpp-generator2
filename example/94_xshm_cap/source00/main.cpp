#include <X11/Xlib.h>
#include <X11/extensions/XShm.h>
#include <cassert>
#include <iostream>
#include <spdlog/spdlog.h>
int main(int argc, char **argv) {
  (void)argv;
  spdlog::info("start  argc='{}'", argc);
  auto *display = XOpenDisplay(nullptr);
  assert(display);
  auto screenNum = DefaultScreen(display);
  auto rootWindow = RootWindow(display, screenNum);
  auto info = XShmSegmentInfo();
  auto w = DisplayWidth(display, screenNum);
  auto h = DisplayHeight(display, screenNum);
  info.shmid = -1;
  auto *image =
      XShmCreateImage(display, nullptr, 24, ZPixmap, nullptr, &info, w, h);
}