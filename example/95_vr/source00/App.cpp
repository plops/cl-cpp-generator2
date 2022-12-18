// no preamble
#include "App.h"
#include "VrApi.h"
#include "VrApi_Helpers.h"
#include "VrApi_Input.h"
#include "VrApi_SystemUtils.h"
#include "android_native_app_glue.h"
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <android/log.h>
#include <android/window.h>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <vector>
App::App(ovrJava *java)
    : java(java), resumed(false), egl(Egl()),
      renderer(
          Renderer(vrapi_GetSystemPropertyInt(
                       java, VRAPI_SYS_PROP_SUGGESTED_EYE_TEXTURE_WIDTH),
                   vrapi_GetSystemPropertyInt(
                       java, VRAPI_SYS_PROP_SUGGESTED_EYE_TEXTURE_HEIGHT))),
      window(nullptr), ovr(nullptr), back_button_down_previous_frame(false),
      frame_index(0) {}
void App::update_vr_mode() {
  if (((resumed) && ((nullptr) != (window)))) {
    if ((nullptr) == (ovr)) {
      auto mode_parms = vrapi_DefaultModeParms(java);
      mode_parms.Flags = ((mode_parms.Flags) | (VRAPI_MODE_FLAG_NATIVE_WINDOW));
      mode_parms.Flags =
          ((mode_parms.Flags) & (~VRAPI_MODE_FLAG_RESET_WINDOW_FULLSCREEN));
      mode_parms.Display = reinterpret_cast<size_t>(egl.display);
      mode_parms.WindowSurface = reinterpret_cast<size_t>(window);
      mode_parms.ShareContext = reinterpret_cast<size_t>(egl.context);
      __android_log_print(ANDROID_LOG_VERBOSE, "hello_quest", "enter vr mode");
      ovr = vrapi_EnterVrMode(&mode_parms);
      if ((nullptr) == (ovr)) {
        __android_log_print(ANDROID_LOG_VERBOSE, "hello_quest",
                            "error: cant enter vr mode");
        std::exit(-1);
      }
      vrapi_SetClockLevels(ovr, CPU_LEVEL, GPU_LEVEL);
    }
  } else {
    if (!((nullptr) == (ovr))) {
      __android_log_print(ANDROID_LOG_VERBOSE, "hello_quest", "leave vr mode");
      vrapi_LeaveVrMode(ovr);
      ovr = nullptr;
    }
  }
}