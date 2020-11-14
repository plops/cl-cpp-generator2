
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <iostream>
#include <thread>
#define SK_GL
#define GR_GL_LOG_CALLS 0
#define GR_GL_CHECK_ERROR 0
#include <GL/gl.h>
#include <SDL2/SDL.h>
#include <include/core/SkCanvas.h>
#include <include/core/SkGraphics.h>
#include <include/core/SkSurface.h>
#include <include/gpu/GrBackendSurface.h>
#include <include/gpu/GrDirectContext.h>
#include <include/gpu/gl/GrGLInterface.h>
#include <src/gpu/gl/GrGLUtil.h>

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

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("") << (" ")
      << (std::setw(8)) << (" SDL_GetPixelFormatName(windowFormat)='")
      << (SDL_GetPixelFormatName(windowFormat)) << ("'") << (std::endl)
      << (std::flush);
  glViewport(0, 0, dw, dh);
  glClearColor(1, 0, 1, 1);
  glClearStencil(0);
  glClear(((GL_COLOR_BUFFER_BIT) | (GL_STENCIL_BUFFER_BIT)));
  SDL_GL_SetSwapInterval(1);
  auto interface = GrGLMakeNativeInterface();
  auto grContext = GrDirectContext::MakeGL(interface);
  SkASSERT(grContext);
  auto buffer = GrGLint(0);
  GR_GL_GetIntegerv(interface.get(), GR_GL_FRAMEBUFFER_BINDING, &buffer);

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("") << (" ")
      << (std::setw(8)) << (" buffer='") << (buffer) << ("'") << (std::endl)
      << (std::flush);
  auto info = GrGLFramebufferInfo();
  info.fFBOID = static_cast<GrGLuint>(buffer);
  auto target = GrBackendRenderTarget(dw, dh, 4, 8, info);
  auto props = SkSurfaceProps();
  auto surface = SkSurface::MakeFromBackendRenderTarget(
      grContext.get(), target, kBottomLeft_GrSurfaceOrigin,
      kRGBA_8888_SkColorType, nullptr, &props);

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("") << (" ")
      << (std::setw(8)) << (" surface='") << (surface) << ("'") << (std::endl)
      << (std::flush);
  for (auto i = 0; (i) < (((60) * (3))); (i) += (1)) {
    glClear(GL_COLOR_BUFFER_BIT);
    SDL_GL_SwapWindow(window);
  }

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("shutdown") << (" ")
      << (std::endl) << (std::flush);

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("destroy gl ctx")
      << (" ") << (std::endl) << (std::flush);
  if (ctx) {
    SDL_GL_DeleteContext(ctx);
  }

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("destroy window")
      << (" ") << (std::endl) << (std::flush);
  SDL_DestroyWindow(window);

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("quit") << (" ")
      << (std::endl) << (std::flush);
  SDL_Quit();
  return 0;
}