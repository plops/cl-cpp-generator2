// no preamble
static const auto initializedSDL = SDL_Init(SDL_INIT_VIDEO);
static constexpr auto TexttureFormat = SDL_PIXEL_FORMAT_ARGB8888;

static constexpr bool successful(int Code) { return 0 == Code; }

#include "FancyWindow.h"
FancyWindow::FancyWindow(tDimensions Dimensions) noexcept {}

bool isAlive() noexcept {
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    if (SDL_QUIT == event.type) {
      return false;
    }
  }
  return true;
}
