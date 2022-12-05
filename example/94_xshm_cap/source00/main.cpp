#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XShm.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <spdlog/spdlog.h>
#include <sys/shm.h>
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
  info.shmid = shmget(IPC_PRIVATE, ((image->bytes_per_line) * (image->height)),
                      ((IPC_CREAT) | (0777)));
  assert((0) <= (info.shmid));
  image->data = reinterpret_cast<char *>(shmat(info.shmid, 0, 0));
  info.shmaddr = image->data;
  info.readOnly = False;
  XShmAttach(display, &info);
  XShmGetImage(display, rootWindow, image, 0, 0, AllPlanes);
  auto file = std::ofstream("screenshot.pgm", std::ios::binary);
  (file) << ("P5") << (std::endl);
  (file) << (image->width) << (" ") << (image->height) << (std::endl);
  (file) << (255) << (std::endl);
  for (auto y = 0; (y) < (image->height); (y) += (1)) {
    for (auto x = 0; (x) < (image->width); (x) += (1)) {
      auto pixel = XGetPixel(image, x, y);
      file.put(static_cast<unsigned char>(pixel));
    }
  }
  file.close();
  shmdt(info.shmaddr);
  shmctl(info.shmid, IPC_RMID, 0);
  XDestroyImage(image);
  XCloseDisplay(display);
  return 0;
}