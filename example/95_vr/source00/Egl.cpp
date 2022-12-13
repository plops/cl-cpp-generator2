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
}