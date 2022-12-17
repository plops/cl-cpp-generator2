#ifndef RENDERER_H
#define RENDERER_H

#include "Program.h"
#include "Geometry.h"
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
class Renderer  {
        public:
        Program program;
        Geometry geometry;
         Renderer (GLsizei width, GLsizei height)     ;  
};

#endif /* !RENDERER_H */