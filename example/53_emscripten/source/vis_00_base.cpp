
#include "utils.h"

#include "globals.h"

#include <chrono>
#include <iostream>
#include <thread>

#include <SDL/SDL.h>
#include <complex>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

using namespace std::chrono_literals;

State state = {};
int main(int argc, char **argv) {
  state._main_version = "202be4ce2980ceace1efba8a88b1b9d2ea5d63a9";
  state._code_repository = "https://github.com/plops/cl-cpp-generator2/tree/"
                           "master/example/53_emscripten/source/";
  state._code_generation_time = "01:33:20 of Friday, 2020-12-11 (GMT+1)";
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
  SDL_CreateWindowAndRenderer(window, height, SDL_WINDOW_OPENGL, &window,
                              &renderer);
  return 0;
}