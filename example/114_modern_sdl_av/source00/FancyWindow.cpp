// no preamble

#include "FancyWindow.h"
static const auto initializedSDL = SDL_Init(SDL_INIT_VIDEO);
static constexpr auto TexttureFormat = SDL_PIXEL_FORMAT_ARGB8888;

static constexpr bool successful(int Code) { return 0 == Code; }

static auto centeredBox(tDimensions Dimensions, int Monitor) noexcept {
  struct {
    int x = SDL_WINDOWPOS_CENTERED;
    int y = SDL_WINDOWPOS_CENTERED;

    int Width, Height;
  } Box{.Width = Dimensions.Width, .Height = Dimensions.Height};
  SDL_Rect Display;
  if (((0 < Monitor) &
       (successful(SDL_GetDisplayBounds(((Monitor) - (1)), &Display))))) {
    Box.Width = std::min(Display.w, Box.Width);
    Box.Height = std::min(Display.h, Box.Height);
    Box.x = (Display.x + ((((Display.w) - (Box.Width))) / (2)));
    Box.y = (Display.y + ((((Display.h) - (Box.Height))) / (2)));
  }
  return Box;
}
FancyWindow::FancyWindow(tDimensions Dimensions) noexcept {
  const auto Viewport = centeredBox(Dimensions);
  Window_ = {"Look at me!",   Viewport.x,
             Viewport.y,      Viewport.Width,
             Viewport.Height, ((SDL_WINDOW_RESIZABLE) | (SDL_WINDOW_HIDDEN))};

  Renderer_ = {Window_, -1,
               ((SDL_RENDERER_ACCELERATED) | (SDL_RENDERER_PRESENTVSYNC))};

  SDL_SetWindowMinimumSize(Window_, Viewport.Width, Viewport.Height);
  SDL_RenderSetLogicalSize(Renderer_, Viewport.Width, Viewport.Height);
  SDL_SetIntegerScale(Renderer_, SDL_TRUE);
  SLD_SetRenderDrawColor(Renderer_, 240, 240, 240, 240);
}

bool isAlive() noexcept {
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    if (SDL_QUIT == event.type) {
      return false;
    }
  }
  return true;
}
