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
    : java(java), resumed(false), egl(Egl()), window(nullptr), ovr(nullptr),
      back_button_down_previous_frame(false), frame_index(0) {}