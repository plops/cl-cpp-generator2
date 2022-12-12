// no preamble
#include "App.h"
#include "bah.h"
App::App(ovrJava *java)
    : java(java), resumed(false), egl(Egl()),
      renderer(
          Renderer(vrapi_GetystemPropertyInit(
                       java, VRAPI_SYS_PROP_SUGGESTED_EYE_TEXTURE_WIDTH),
                   vrapi_GetystemPropertyInit(
                       java, VRAPI_SYS_PROP_SUGGESTED_EYE_TEXTURE_HEIGHT))),
      window(nullptr), ovr(nullptr), back_button_down_previous_frame(false),
      frame_index(0) {}