// no preamble
#include "AttribPointer.h"
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
AttribPointer::AttribPointer(GLint size, GLenum type, GLboolean normalized,
                             GLsizei stride, const GLvoid *pointer)
    : size(size), type(type), normalized(normalized), stride(stride),
      pointer(pointer) {}