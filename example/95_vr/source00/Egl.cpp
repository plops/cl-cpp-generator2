// no preamble
#include "Egl.h"
#include "bah.h"
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
    if ((0) == (((renderable_type) & (EGL_OPENGL_ES3_BIT_KHR)))) {
      continue;
    }
    if ((0) == (((renderable_type) & (EGL_PBUFFER_BIT)))) {
      continue;
    }
    if ((0) == (((renderable_type) & (EGL_WINDOW_BIT)))) {
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
  }
}