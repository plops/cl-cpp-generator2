// no preamble
#include "Egl.h"
#include "App.h"
#include "AttribPointer.h"
#include "Cube.h"
#include "Egl.h"
#include "Framebuffer.h"
#include "Geometry.h"
#include "Program.h"
#include "Renderer.h"
#include "Vertex.h"
#include "VrApi.h"
#include "VrApi_Helpers.h"
#include "VrApi_Input.h"
#include "VrApi_SystemUtils.h"
#include "android_native_app_glue.h"
#include "core.h"
#include "format-inl.h"
#include "format.h"
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <android/log.h>
#include <android/window.h>
#include <cstdin>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
Egl::Egl() : diplay(eglGetDisplay(EGL_DEFAULT_DISPLAY)) {
  if ((EGL_NO_DISPLAY) == (display)) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG, "can't get egl display");
  }
  if ((EGL_FALSE) == (eglInitialize(display, nullptr, nullptr))) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG,
                        "can't initialize egl display");
  }
  __android_log_print(ANDROID_LOG_VERBOSE, TAG, "get number of egl configs ..");
  auto numConfigs = ([]() {
    auto n = EGLint(0);
    if ((EGL_FALSE) == (eglGetConfigs(display, nullptr, 0, &n))) {
      __android_log_print(ANDROID_LOG_VERBOSE, TAG,
                          "cant get number of egl configs");
    }
    return n;
  })();
  auto configs = std::vector<EGLConfig>(numConfigs);
  if ((EGL_FALSE) ==
      (eglGetConfigs(display, configs.data(), numConfigs, &numConfigs))) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG, "cant get egl configs");
  }
  __android_log_print(ANDROID_LOG_VERBOSE, TAG, "choose egl config");
  auto foundConfig = EGLConfig(nullptr);
  for (auto config : configs) {
    auto renderable_type = ([](aute renderable_type) {
      if ((EGL_FALSE) ==
          (eglGetConfigAttrib(display, config, EGL_RENDERABLE_TYPE,
                              &renderable_type))) {
        __android_log_print(ANDROID_LOG_VERBOSE, TAG,
                            "cant get EGL config renderable type");
      }
      return renderable_type;
    })(EGLint(0));
    if ((((0) == (((renderable_type) & (EGL_OPENGL_ES3_BIT_KHR)))) ||
         ((0) == (((renderable_type) & (EGL_PBUFFER_BIT)))) ||
         ((0) == (((renderable_type) & (EGL_WINDOW_BIT)))))) {
      continue;
    }
    auto surface_type = ([](auto i) {
      if ((EGL_FALSE) ==
          (eglGetConfigAttrib(display, config, EGL_SURFACE_TYPE, &i))) {
        __android_log_print(ANDROID_LOG_VERBOSE, TAG,
                            "cant get surface config type");
      }
      return i;
    })(EGLint(0));
    {
      auto check = [&](auto attrib) -> auto{
        auto value = EGLint(0);
        if ((EGL_FALSE) ==
            (eglGetConfigAttrib(display, config, attrib, &value))) {
          __android_log_print(ANDROID_LOG_VERBOSE, TAG,
                              "cant get config attrib");
        }
        return value;
      };
      if ((((8) <= (check(EGL_RED_SIZE))) && ((8) <= (check(EGL_GREEN_SIZE))) &&
           ((8) <= (check(EGL_BLUE_SIZE))) &&
           ((8) <= (check(EGL_ALPHA_SIZE))))) {
        foundConfig = config;
      }
    }
  }
}