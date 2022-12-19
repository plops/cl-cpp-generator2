#ifndef EGL_H
#define EGL_H

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
#include <array>
#include <string_view>
#include "DataTypes.h"
class Egl  {
        public:
        EGLDisplay display;
        EGLContext context;
        EGLSurface surface;
        std::string_view egl_get_error_string (EGLint err)     ;  
         Egl ()     ;  
};

#endif /* !EGL_H */