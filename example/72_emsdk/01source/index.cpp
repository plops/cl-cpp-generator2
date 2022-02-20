// no preamble
;
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
extern std::mutex g_stdout_mutex;
extern std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
#include "index.h"
std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
std::mutex g_stdout_mutex;
extern "C" int main(int argc, char **argv) {
  g_start_time = std::chrono::high_resolution_clock::now();
  {
    {

      auto lock = std::unique_lock<std::mutex>(g_stdout_mutex);
      std::chrono::duration<double> timestamp =
          std::chrono::high_resolution_clock::now() - g_start_time;
      (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                  << (std::this_thread::get_id()) << (" ") << (__FILE__)
                  << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                  << ("enter program") << (" ") << (std::setw(8)) << (" argc='")
                  << (argc) << ("'") << (std::setw(8)) << (" argv='") << (argv)
                  << ("'") << (std::endl) << (std::flush);
    }
    SDL_Init(SDL_INIT_VIDEO);
    auto screen = SDL_SetVideoMode(256, 256, 32, SDL_SWSURFACE);
    if (SDL_MUSTLOCK(screen)) {
      {

        auto lock = std::unique_lock<std::mutex>(g_stdout_mutex);
        std::chrono::duration<double> timestamp =
            std::chrono::high_resolution_clock::now() - g_start_time;
        (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                    << (std::this_thread::get_id()) << (" ") << (__FILE__)
                    << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                    << ("lock screen") << (" ") << (std::endl) << (std::flush);
      }
      SDL_LockSurface(screen);
    }
    {

      auto lock = std::unique_lock<std::mutex>(g_stdout_mutex);
      std::chrono::duration<double> timestamp =
          std::chrono::high_resolution_clock::now() - g_start_time;
      (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                  << (std::this_thread::get_id()) << (" ") << (__FILE__)
                  << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                  << ("draw") << (" ") << (std::endl) << (std::flush);
    }
    for (auto i = 0; (i) < (256); (i) += (1)) {
      for (auto j = 0; (j) < (256); (j) += (1)) {
        auto alpha = 255;
        static_cast<Uint32 *>(screen->pixels)[((i) + (((256) * (j))))] =
            SDL_MapRGBA(screen->format, i, j, ((255) - (i)), alpha);
      }
    }
    if (SDL_MUSTLOCK(screen)) {
      {

        auto lock = std::unique_lock<std::mutex>(g_stdout_mutex);
        std::chrono::duration<double> timestamp =
            std::chrono::high_resolution_clock::now() - g_start_time;
        (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                    << (std::this_thread::get_id()) << (" ") << (__FILE__)
                    << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                    << ("unlock screen") << (" ") << (std::endl)
                    << (std::flush);
      }
      SDL_UnlockSurface(screen);
    }
    {

      auto lock = std::unique_lock<std::mutex>(g_stdout_mutex);
      std::chrono::duration<double> timestamp =
          std::chrono::high_resolution_clock::now() - g_start_time;
      (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                  << (std::this_thread::get_id()) << (" ") << (__FILE__)
                  << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                  << ("flip screen") << (" ") << (std::endl) << (std::flush);
    }
    SDL_Flip(screen);
    {

      auto lock = std::unique_lock<std::mutex>(g_stdout_mutex);
      std::chrono::duration<double> timestamp =
          std::chrono::high_resolution_clock::now() - g_start_time;
      (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                  << (std::this_thread::get_id()) << (" ") << (__FILE__)
                  << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                  << ("quit sdl") << (" ") << (std::endl) << (std::flush);
    }
    SDL_Quit();
    {

      auto lock = std::unique_lock<std::mutex>(g_stdout_mutex);
      std::chrono::duration<double> timestamp =
          std::chrono::high_resolution_clock::now() - g_start_time;
      (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                  << (std::this_thread::get_id()) << (" ") << (__FILE__)
                  << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                  << ("exit program") << (" ") << (std::endl) << (std::flush);
    }
    return 0;
  }
}