#include "App.h"
#include "Vertex.h"
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
const std::array<AttribPointer, 2> ATTRIB_POINTERS = {
    AttribPointer(3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                  reinterpret_cast<GLvoid *>(offsetof(Vertex, position))),
    AttribPointer(3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                  reinterpret_cast<GLvoid *>(offsetof(Vertex, color)))};
void app_on_cmd(android_app *android_app, int32_t cmd) {
  auto app = reinterpret_cast<App *>(android_app->userData);
  switch (cmd) {
  case APP_CMD_START: {
    __android_log_print(ANDROID_LOG_VERBOSE, "hello_quest", "(start)");
    break;
  }
  case APP_CMD_RESUME: {
    __android_log_print(ANDROID_LOG_VERBOSE, "hello_quest",
                        "(resume (setf app->resumed true))");
    app->resumed = true;
    break;
  }
  case APP_CMD_PAUSE: {
    __android_log_print(ANDROID_LOG_VERBOSE, "hello_quest",
                        "(pause (setf app->resumed false))");
    app->resumed = false;
    break;
  }
  case APP_CMD_STOP: {
    __android_log_print(ANDROID_LOG_VERBOSE, "hello_quest", "(stop)");
    break;
  }
  case APP_CMD_DESTROY: {
    __android_log_print(ANDROID_LOG_VERBOSE, "hello_quest",
                        "(destroy (setf app->window nullptr))");
    app->window = nullptr;
    break;
  }
  case APP_CMD_INIT_WINDOW: {
    __android_log_print(ANDROID_LOG_VERBOSE, "hello_quest",
                        "(init-window (setf app->window android_app->window))");
    app->window = android_app->window;
    break;
  }
  case APP_CMD_TERM_WINDOW: {
    __android_log_print(ANDROID_LOG_VERBOSE, "hello_quest",
                        "(term-window (setf app->window nullptr))");
    app->window = nullptr;
    break;
  }
  default: {
    __android_log_print(ANDROID_LOG_VERBOSE, "hello_quest",
                        "app_on_cmd default");
    break;
  }
  }
}
void android_main(android_app *android_app) {
  ANativeActivity_setWindowFlags(android_app->activity,
                                 AWINDOW_FLAG_KEEP_SCREEN_ON, 0);
  __android_log_print(ANDROID_LOG_VERBOSE, "hello_quest",
                      "attach current thread");
  auto java = ovrJava();
  java.Vm = android_app->activity->vm;
  java.Vm->AttachCurrentThread(&java.Env, nullptr);
  java.ActivityObject = android_app->activity->clazz;
  __android_log_print(ANDROID_LOG_VERBOSE, "hello_quest", "initialize vr api");
  auto init_params = vrapi_DefaultInitParms(&java);
  if (!((VRAPI_INITIALIZE_SUCCESS) == (vrapi_Initialize(&init_params)))) {
    __android_log_print(ANDROID_LOG_VERBOSE, "hello_quest",
                        "can't initialize vr api");
    std::exit(1);
  }
  auto app = App(&java);
  android_app->userData = &(app);
  android_app->onAppCmd = app_on_cmd;
  while (!(android_app->destroyRequested)) {
    while (true) {
      auto events = 0;
      android_poll_source *source = nullptr;
      if ((ALooper_pollAll(((android_app->destroyRequested) ||
                            (((nullptr) != (app.ovr)) ? (0) : (-1))),
                           nullptr, &events,
                           reinterpret_cast<void **>(&source))) < (0)) {
        break;
      }
      if (!((nullptr) == (source))) {
        source->process(android_app, source);
      }
      app.update_vr_mode();
    }
    app.handle_input();
    if ((nullptr) == (app.ovr)) {
      continue;
    }
    (app.frame_index)++;
    auto display_time = vrapi_GetPredictedDisplayTime(app.ovr, app.frame_index);
    auto tracking = vrapi_GetPredictedTracking2(app.ovr, display_time);
    auto layer = app.renderer.render_frame(&tracking);
    auto layers = std::array<ovrLayerHeader2 *, 1>({&layer.Header});
    ovrSubmitFrameDescription2 frame = {.Flags = 0,
                                        .SwapInterval = 1,
                                        .FrameIndex = app.frame_index,
                                        .DisplayTime = display_time,
                                        .LayerCount = 1,
                                        .Layers = layers.data()};
    vrapi_SubmitFrame2(app.ovr, &frame);
  }
  __android_log_print(ANDROID_LOG_VERBOSE, "hello_quest", "shut down vr api");
  vrapi_Shutdown();
  __android_log_print(ANDROID_LOG_VERBOSE, "hello_quest",
                      "detach current thread");
  java.Vm->DetachCurrentThread();
}