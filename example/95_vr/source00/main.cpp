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
#include <cstdin>
#include <cstdlib>
#include <iostream>
#include <spdlog/spdlog.h>
#include <unistd.h>
#define FMT_HEADER_ONLY
#include "core.h"
static const std::array<AttribPointer, 2> ATTRIB_POINTERS = {
    AttribPointer(3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                  reinterpret_cast<GLvoid *>(offsetof(Vertex, position))),
    AttribPointer(3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                  reinterpret_cast<GLvoid *>(offsetof(Vertex, color)))};
void android_main(android_app *android_app) {
  ANativeActivity_setWindowFlags(android_app->activity,
                                 AWINDOW_FLAG_KEEP_SCREEN_ON, 0);
  __android_log_print(ANDROID_LOG_VERBOSE, TAG, "attach current thread");
  auto java = ovrJava();
  java.Vm = android_app->activity->vm;
  *(java.Vm)->AttachCurrentThread(java.Vm, &java.Env, nullptr);
  __android_log_print(ANDROID_LOG_VERBOSE, TAG, "initialize vr api");
  auto init_params = vrapi_DefaultInitParams(&java);
  if (!((VRAPI_INITIALIZE_SUCCESS) == (vrapi_Initialize(&init_params)))) {
    __android_log_print(ANDROID_LOG_VERBOSE, TAG, "can't initialize vr api");
    std::exit(1);
  }
  auto app = App(&app, &java);
}