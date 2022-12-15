// no preamble
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
App::App(ovrJava *java)
    : java(java), resumed(false), egl(Egl()),
      renderer(
          Renderer(vrapi_GetystemPropertyInit(
                       java, VRAPI_SYS_PROP_SUGGESTED_EYE_TEXTURE_WIDTH),
                   vrapi_GetystemPropertyInit(
                       java, VRAPI_SYS_PROP_SUGGESTED_EYE_TEXTURE_HEIGHT))),
      window(nullptr), ovr(nullptr), back_button_down_previous_frame(false),
      frame_index(0) {}