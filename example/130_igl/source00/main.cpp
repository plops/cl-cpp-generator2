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
int width_ = 800;
int height_ = 600;
std::unique_ptr<IDevice> device_;
std::shared_ptr<ICommandQueue> commandQueue_;
RenderPassDesc renderPass_;
std::shared_ptr<IFramebuffer> framebuffer_;
std::shared_ptr<IRenderPipelineState> renderPipelineState_Triangle_;

static bool initWindow(GLFWwindow **outWindow) {
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
  auto *window =
      glfwCreateWindow(800, 600, "OpenGL Triangle", nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    return false;
  }
  glfwSetErrorCallback([](int err, const char *desc) {
    std::cout << "GLFW Error"
              << " err='" << err << "' "
              << " desc='" << desc << "' " << std::endl;
  });
  glfwSetKeyCallback(window,
                     [](GLFWwindow *window, int key, int a, int action, int b) {
                       if (key == GLFW_KEY_ESCAPE & action == GLFW_PRESS) {
                         glfwSetWindowShouldClose(window, GLFW_TRUE);
                       }
                     });
  glfwSetWindowSizeCallback(
      window, [](GLFWwindow *window, int width, int height) {
        std::cout << "window resized"
                  << " width='" << width << "' "
                  << " height='" << height << "' " << std::endl;
        width_ = width;
        height_ = height;
      });
  glfwGetWindowSize(window, &width_, &height_);
  if (outWindow) {
    *outWindow = window;
  }
  return true;
}

void initGL() {
  auto ctx = std::make_unique<igl::opengl::glx::Context>(
      nullptr, glfwGetX11Display(),
      reinterpret_cast<igl::opengl::glx::GLXDrawable>(
          glfwGetX11Window(window_)),
      reinterpret_cast<igl::opengl::glx::GLXContext>(
          glfwGetGLXContext(window_)));
  device_ = std::make_unique<igl::opengl::glx::Device>(std::move(ctx));

  IGL_ASSERT(device_);
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  return 0;
}
