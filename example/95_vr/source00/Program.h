#ifndef PROGRAM_H
#define PROGRAM_H

#pragma once
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
#include "DataTypes.h"
#include "DataExtern.h"
#include <array>
class Program  {
        public:
        GLuint program;
        std::array<GLint,UNIFORM_END> uniform_locations;
        GLuint compileShader (GLenum type, std::string str)     ;  
         Program ()     ;  
};

#endif /* !PROGRAM_H */