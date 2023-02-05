// no preamble
bool isAlive() noexcept {
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    if (SDL_QUIT == event.type) {
      return false;
    }
  }
  return true;
};
#include "FancyWindow.h"
FancyWindow::FancyWindow(tDimensions Dimensions) {}
