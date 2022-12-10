#include <iostream>
#include <spdlog/spdlog.h>
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
#include <unistd.h>
void main (android_app* android_app)      {
        __android_log_print(ANDROID_LOG_VERBOSE, TAG, "attach current thread");
}