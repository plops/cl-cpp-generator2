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
using namespace igl;
static const uint32_t kNumColorAttachments = 1;
std::string codeVS = R"(#version 460
layout (location=0) out vec3 color; 
const vec2 pos[3]  = vec2[3](vec2(-0.60f, -0.40f), vec2(0.60f, -0.40f), vec2(0.f, 0.60f)); 
const vec3 col[3]  = vec3[3](vec3(1.0f, 0.f, 0.f), vec3(0.f, 1.0f, 0.f), vec3(0.f, 0.f, 1.0f)); 

void main ()        {
            gl_Position=vec4(pos[gl_VertexIndex], 0.f, 1);
    color=gl_VertexIndex;


}
 
 
)";

std::string codeFS = R"(#version 460
layout (location=0) in vec3 color; 
layout (location=0) out vec4 out_FragColor; 

void main ()        {
            out_FragColor=vec4(color, 1);


}
 
)";

GLFWwindow *window_ = nullptr;
int width_ = 0;
int height_ = 0;
std::unique_ptr<IDevice> device_;
std::shared_ptr<ICommandQueue> commandQueue_;
RenderPassDesc renderPass_;
std::shared_ptr<IFrameBuffer> framebuffer_;
std::shared_ptr<IRenderPipelineState> renderPipelineState_Triangle_;

static bool initWindow(GLFWwindow *outWindow) {
  if (!glfwInit()) {
    return false;
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
  glfwWindowHint(GLFW_VISIBLE, true);
  glfwWindowHint(GLFW_DOUBLEBUFFER, true);
  glfwWindowHint(GLFW_SRGB_CAPABLE, true);
  glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  return 0;
}
