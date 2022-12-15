// no preamble
#include "Cube.h"
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
Cube::Cube()
    : vertices({{-1, 1, -1},
                {1, 0, 1},
                {1, 1, -1},
                {0, 1, 0},
                {1, 1, 1},
                {0, 0, 1},
                {-1, 1, 1},
                {1, 0, 0},
                {-1, -1, -1},
                {0, 0, 1},
                {-1, -1, 1},
                {0, 1, 0},
                {1, -1, 1},
                {1, 0, 1},
                {1, -1, -1},
                {1, 0, 0}}),
      indices({0, 1, 2, 2, 0, 3, 4, 6, 5, 6, 4, 7, 2, 6, 7, 7, 1, 2,
               0, 4, 5, 5, 3, 0, 3, 5, 6, 6, 2, 3, 0, 1, 7, 7, 4, 0}) {}