// no preamble

#include "FancyWindow.h"
static const auto initializedSDL = SDL_Init(SDL_INIT_VIDEO);
static constexpr auto TextureFormat = SDL_PIXELFORMAT_ARGB8888;

static constexpr bool successful(int Code) { return 0 == Code; }

static auto centeredBox(tDimensions Dimensions, int Monitor) noexcept {
  struct {
    int x = SDL_WINDOWPOS_CENTERED;
    int y = SDL_WINDOWPOS_CENTERED;

    int Width, Height;
  } Box{.Width = Dimensions.Width, .Height = Dimensions.Height};
  SDL_Rect Display;
  if (((0 < Monitor) &&
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
  // SDL_Window * SDL_CreateWindow(const char *title, int x, int y, int w, int
  // h, Uint32 flags);
  Window_ = {"Look at me!",   Viewport.x,
             Viewport.y,      Viewport.Width,
             Viewport.Height, ((SDL_WINDOW_RESIZABLE) | (SDL_WINDOW_HIDDEN))};

  // SDL_Renderer * SDL_CreateRenderer(SDL_Window * window, int index, Uint32
  // flags);
  Renderer_ = {Window_, -1,
               ((SDL_RENDERER_ACCELERATED) | (SDL_RENDERER_PRESENTVSYNC))};

  SDL_SetWindowMinimumSize(Window_, Viewport.Width, Viewport.Height);
  SDL_RenderSetLogicalSize(Renderer_, Viewport.Width, Viewport.Height);
  SDL_RenderSetIntegerScale(Renderer_, SDL_TRUE);
  SDL_SetRenderDrawColor(Renderer_, 240, 240, 240, 240);
}
void FancyWindow::updateFrom() noexcept {
  Width_ = 320;
  Height_ = 240;
  PixelsPitch_ = 240;
  SourceFormat_ = SDL_PIXELFORMAT_ARGB8888;

  Texture_ = Texture(Renderer_, TextureFormat, SDL_TEXTUREACCESS_STREAMING,
                     Width_, Height_);

  SDL_SetWindowMinimumSize(Window_, Width_, Height_);
  SDL_RenderSetLogicalSize(Renderer_, Width_, Height_);
  SDL_ShowWindow(Window_);
}
void FancyWindow::present() noexcept {
  SDL_RenderClear(Renderer_);
  void *TextureData;
  int TexturePitch;
  if (successful(
          SDL_LockTexture(Texture_, nullptr, &TextureData, &TexturePitch))) {
    SDL_UnlockTexture(Texture_);
    SDL_RenderCopy(Renderer_, Texture_, nullptr, nullptr);
  }
  SDL_RenderPresent(Renderer_);
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
