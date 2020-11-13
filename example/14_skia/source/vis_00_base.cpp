
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <iostream>
#include <thread>
#define SK_GL
#include <GL/gl.h>
#include <SDL2/SDL.h>
#include <include/core/SkCanvas.h>
#include <include/core/SkGraphics.h>
#include <include/core/SkSurface.h>
#include <include/gpu/GrBackendSurface.h>
#include <include/gpu/GrDirectContext.h>
#include <include/gpu/gl/GrGLInterface.h>

// implementation
int main(int argc, char **argv) {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("start") << (" ")
      << (std::setw(8)) << (" argc='") << (argc) << ("'") << (std::setw(8))
      << (" argv[0]='") << (argv[0]) << ("'") << (std::endl) << (std::flush);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  auto windowFlags = ((SDL_WINDOW_OPENGL) | (SDL_WINDOW_RESIZABLE));
  SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
  SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
  SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 0);
  SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
  SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1);
  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);
  if (!((0) == (SDL_Init(((SDL_INIT_VIDEO) | (SDL_INIT_EVENTS)))))) {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("init error") << (" ") << (std::endl) << (std::flush);
  }
  auto window = SDL_CreateWindow("sdl window", SDL_WINDOWPOS_CENTERED,
                                 SDL_WINDOWPOS_CENTERED, 512, 200, windowFlags);
  if (!(window)) {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("window error") << (" ") << (std::endl) << (std::flush);
  }
  auto ctx = SDL_GL_CreateContext(window);
  if (!(ctx)) {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("ctx error") << (" ") << (std::endl) << (std::flush);
  }
  auto success = SDL_GL_MakeCurrent(window, ctx);
  if (!((0) == (success))) {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("makecurrent error") << (" ") << (std::endl)
                << (std::flush);
  }
  auto windowFormat = SDL_GetWindowPixelFormat(window);
  auto dw = int(0);
  auto dh = dw;
  auto contextType = dw;
  SDL_GL_GetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, &contextType);
  SDL_GL_GetDrawableSize(window, &dw, &dh);

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("") << (" ")
      << (std::setw(8)) << (" windowFormat='") << (windowFormat) << ("'")
      << (std::setw(8)) << (" contextType='") << (contextType) << ("'")
      << (std::setw(8)) << (" dw='") << (dw) << ("'") << (std::setw(8))
      << (" dh='") << (dh) << ("'") << (std::endl) << (std::flush);
  glViewport(0, 0, dw, dh);
  glClearColor(1, 1, 1, 1);
  glClearStencil(0);
  glClear(((GL_COLOR_BUFFER_BIT) | (GL_STENCIL_BUFFER_BIT)));
  auto interface = GrGLMakeNativeInterface();
  auto grContext = sk_sp<GrDirectContext>(GrDirectContext::MakeGL());
  SkASSERT(grContext);
  SDL_DestroyWindow(window);
  SDL_Quit();
  return 0;
}