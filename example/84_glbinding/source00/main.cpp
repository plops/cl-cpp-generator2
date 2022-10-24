#include <chrono>
#include <glbinding/AbstractFunction.h>
#include <glbinding/CallbackMask.h>
#include <glbinding/FunctionCall.h>
#include <glbinding/gl32core/gl.h>
#include <glbinding/glbinding.h>
#include <iomanip>
#include <iostream>
#include <thread>
using namespace gl32core;
using namespace glbinding;
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
void lprint(std::initializer_list<std::string> il) {
  std::chrono::duration<double> timestamp(0);
  timestamp = ((std::chrono::high_resolution_clock::now()) - (g_start_time));
  const auto defaultWidth = 10;
  (std::cout) << (std::setw(defaultWidth)) << (timestamp.count()) << (" ")
              << (std::this_thread::get_id()) << (" ");
  for (const auto &elem : il) {
    (std::cout) << (elem);
  }
  (std::cout) << (std::endl) << (std::flush);
}
int main(int argc, char **argv) {
  g_start_time = std::chrono::high_resolution_clock::now();
  lprint({__FILE__, ":", std::to_string(__LINE__), " ", __PRETTY_FUNCTION__,
          " ", "start", " ", " argc='", std::to_string(argc), "'"});
  auto *window = ([]() -> GLFWwindow * {
    if (!(glfwInit())) {
      lprint({__FILE__, ":", std::to_string(__LINE__), " ", __PRETTY_FUNCTION__,
              " ", "glfwInit failed", " "});
    }
    const auto startWidth = 800;
    const auto startHeight = 600;
    auto window = glfwCreateWindow(startWidth, startHeight, "hello bgfx",
                                   nullptr, nullptr);
    if (!(window)) {
      lprint({__FILE__, ":", std::to_string(__LINE__), " ", __PRETTY_FUNCTION__,
              " ", "can't create glfw window", " "});
    }
    glfwMakeContextCurrent(window);
    return window;
  })();
  auto width = int(0);
  auto height = int(0);
  // if second arg is false: lazy function pointer loading
  ;
  glbinding::initialize(glfwGetProcAddress, false);
  {
    const float r = (0.40f);
    const float g = (0.40f);
    const float b = (0.20f);
    const float a = (1.0f);
    glClearColor(r, g, b, a);
  }
  while (!(glfwWindowShouldClose(window))) {
    glfwPollEvents();
    ([&width, &height, window]() {
      // react to changing window size
      ;
      auto oldwidth = width;
      auto oldheight = height;
      glfwGetWindowSize(window, &width, &height);
      if ((((width) != (oldwidth)) || ((height) != (oldheight)))) {
        // set view
        ;
        glViewport(0, 0, width, height);
      }
    })();
    // draw frame
    ;
    glClear(GL_COLOR_BUFFER_BIT);
    glfwSwapBuffers(window);
  }
  glfwTerminate();
  return 0;
}