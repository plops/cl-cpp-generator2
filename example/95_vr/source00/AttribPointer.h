#ifndef ATTRIBPOINTER_H
#define ATTRIBPOINTER_H

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
class AttribPointer  {
        public:
        GLint size;
        GLenum type;
        GLboolean normalized;
        GLsizei stride;
        const GLvoid* pointer;
         AttribPointer (GLint size, GLenum type, GLboolean normalized, GLsizei stride, const GLvoid* pointer)     ;  
};

#endif /* !ATTRIBPOINTER_H */