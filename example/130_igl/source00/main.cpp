#define GLFW_INCLUDE_NONE ;
#define GLFW_EXPOSE_NATIVE_X11 ;
#define GLFW_EXPOSE_NATIVE_GLX ;
#define USE_OPENGL_BACKEND 1;
#define ENABLE_MULTIPLE_COLOR_ATTACHMENTS 0;
#define IGL_FORMAT fmt::format;
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <cassert>
#include <fmt/core.h>
#include <igl/IGL.h>
#include <igl/opengl/glx/Context.h>
#include <igl/opengl/glx/Device.h>
#include <igl/opengl/glx/HWDevice.h>
#include <igl/opengl/glx/PlatformDevice.h>
#include <iostream>
#include <regex>
static const uint32_t kNumColorAttachments = 1;
(std::string codeVS) = (R"(#version 460)");

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  return 0;
}
