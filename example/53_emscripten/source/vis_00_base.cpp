
#include "utils.h"

#include "globals.h"

#include <chrono>
#include <iostream>
#include <thread>

#include <SDL2/SDL.h>
#include <complex>
#include <unistd.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

using namespace std::chrono_literals;

State state = {};
int main(int argc, char **argv) {
  state._main_version = "da0a98f0e2c923f28b4e5a92144df87a49f95bcf";
  state._code_repository = "https://github.com/plops/cl-cpp-generator2/tree/"
                           "master/example/53_emscripten/source/";
  state._code_generation_time = "02:29:41 of Friday, 2020-12-11 (GMT+1)";
  state._start_time =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  {

    auto lock = std::unique_lock<std::mutex>(state._stdout_mutex);
    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("start main") << (" ") << (std::setw(8))
                << (" state._main_version='") << (state._main_version) << ("'")
                << (std::setw(8)) << (" state._code_repository='")
                << (state._code_repository) << ("'") << (std::setw(8))
                << (" state._code_generation_time='")
                << (state._code_generation_time) << ("'") << (std::endl)
                << (std::flush);
  }
  auto width = 600;
  auto height = 800;
  SDL_Init(SDL_INIT_VIDEO);
  auto window = static_cast<SDL_Window *>(nullptr);
  auto renderer = static_cast<SDL_Renderer *>(nullptr);
  SDL_CreateWindowAndRenderer(width, height, SDL_WINDOW_OPENGL, &window,
                              &renderer);
  enum { MAX_ITER_COUNT = 256 };
  SDL_Color palette[MAX_ITER_COUNT];
  for (auto i = 0; (i) < (MAX_ITER_COUNT); (i) += (1)) {
    palette[i] = {.r = uint8_t(rand()),
                  .g = uint8_t(rand()),
                  .b = uint8_t(rand()),
                  .a = 255};
  }
  auto center = std::complex<double>((0.50), (0.50));
  auto scale = (4.0);
  for (auto y = 0; (y) < (height); (y) += (1)) {
    for (auto x = 0; (x) < (width); (x) += (1)) {
      auto point = std::complex<double>((((((1.0)) * (x))) / (width)),
                                        (((((1.0)) * (y))) / (height)));
      auto c = std::complex<double>(((((point) - (center))) * (scale)));
      auto z = std::complex<double>((0.), (0.));
      auto i = 0;
      for (; (i) < (((MAX_ITER_COUNT) - (1))); (i)++) {
        z = ((((z) * (z))) + (c));
        if (((2.0f)) < (abs(z))) {
          break;
        }
      }
      auto color = palette[i];
      SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
      SDL_RenderDrawPoint(renderer, x, y);
    }
  }
  SDL_RenderPresent(renderer);
  return 0;
}