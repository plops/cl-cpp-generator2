// no preamble
#include "Vertex.h"
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
#include "bah.h"
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
Vertex::Vertex(GLint size, GLenum type, GLboolean normalized, GLsizei stride,
               const GLvoid *pointer)
    : size(size), type(type), normalized(normalized), stride(stride),
      pointer(pointer) {}