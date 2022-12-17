// no preamble
#include "Framebuffer.h"
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
Framebuffer::Framebuffer(GLsizei w, GLsizei h)
    : swap_chain_index(0), width(w), height(h),
      color_texture_swap_chain(vrapi_CreateTextureSwapChain3(
          VRAPI_TEXTURE_TYPE_2D, GL_RGBA8, w, h, 1, 3)) {}