#ifndef VERTEX_H
#define VERTEX_H

#include <array>
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
class Vertex  {
public:
    std::array<float,4> position;
    std::array<float,4> color;
    Vertex (GLint size, GLenum type, GLboolean normalized, GLsizei stride, const GLvoid* pointer)     ;
};

#endif /* !VERTEX_H */