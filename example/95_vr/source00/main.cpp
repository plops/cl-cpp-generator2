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
#define FMT_HEADER_ONLY
#include "core.h"
const std::vector<std::string> ATTRIB_NAMES = {"aPosition", "aColor"};
const std::vector<std::string> UNIFORM_NAMES = {"uModelMatrix", "uViewMatrix",
                                                "uProjectionMatrix"};
void android_main(android_app *android_app) {
  ANativeActivity_setWindowFlags(android_app->activity,
                                 AWINDOW_FLAG_KEEP_SCREEN_ON, 0);
  __android_log_print(ANDROID_LOG_VERBOSE, "hello_quest",
                      "attach current thread");
  auto java = ovrJava();
  java.Vm = android_app->activity->vm;
  java.Vm->AttachCurrentThread(&java.Env, nullptr);
  __android_log_print(ANDROID_LOG_VERBOSE, "hello_quest", "initialize vr api");
  auto init_params = vrapi_DefaultInitParms(&java);
  if (!((VRAPI_INITIALIZE_SUCCESS) == (vrapi_Initialize(&init_params)))) {
    __android_log_print(ANDROID_LOG_VERBOSE, "hello_quest",
                        "can't initialize vr api");
    std::exit(1);
  }
  auto app = App(&java);
}