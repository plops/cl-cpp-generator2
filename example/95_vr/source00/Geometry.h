#ifndef GEOMETRY_H
#define GEOMETRY_H

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
class Geometry  {
        public:
        GLuint vertex_array, vertex_buffer, index_buffer;
         Geometry ()     ;  
         ~Geometry ()     ;  
};

#endif /* !GEOMETRY_H */