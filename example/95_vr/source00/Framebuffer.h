#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <iostream>
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
#include <vector>
#include <cstdlib>
#include <unistd.h>
class Framebuffer  {
        public:
        int swap_chain_index;
        int swap_chain_length;
        GLsizei width;
        GLsizei height;
        ovrTextureSwapChain* color_texture_swap_chain;
        GLuint* depth_renderbuffers;
        GLuint* framebuffers;
         Framebuffer (GLsizei w, GLsizei h)     ;  
};

#endif /* !FRAMEBUFFER_H */