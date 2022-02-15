// no preamble
;
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_opengl3_loader.h"
#include <GLFW/glfw3.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
extern std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
#include "GraphicsFramework.h"
GraphicsFramework::GraphicsFramework() {
  glfwSetErrorCallback([](int err, const char *description) {
    {

      std::chrono::duration<double> timestamp =
          std::chrono::high_resolution_clock::now() - g_start_time;
      (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                  << (std::this_thread::get_id()) << (" ") << (__FILE__)
                  << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                  << ("glfw error") << (" ") << (std::setw(8)) << (" err='")
                  << (err) << ("'") << (std::setw(8)) << (" description='")
                  << (description) << ("'") << (std::endl) << (std::flush);
    }
  });
  // glfw initialization
  ;
  // https://github.com/ocornut/imgui/blob/docking/examples/example_glfw_opengl3/main.cpp
  ;
  if (!(glfwInit())) {
    {

      std::chrono::duration<double> timestamp =
          std::chrono::high_resolution_clock::now() - g_start_time;
      (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                  << (std::this_thread::get_id()) << (" ") << (__FILE__)
                  << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                  << ("glfwInit failed.") << (" ") << (std::endl)
                  << (std::flush);
    }
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  auto w = glfwCreateWindow(1800, 1000, "dear imgui example", nullptr, nullptr);
  if ((nullptr) == (w)) {
    {

      std::chrono::duration<double> timestamp =
          std::chrono::high_resolution_clock::now() - g_start_time;
      (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                  << (std::this_thread::get_id()) << (" ") << (__FILE__)
                  << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                  << ("glfwCreatWindow failed.") << (" ") << (std::endl)
                  << (std::flush);
    }
  }
  window.reset(w, [](GLFWwindow *ptr) { glfwDestroyWindow(ptr); });
  glfwMakeContextCurrent(window.get());
  {

    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
                << (__LINE__) << (" ") << (__func__) << (" ")
                << ("enable vsync") << (" ") << (std::endl) << (std::flush);
  }
  glfwSwapInterval(1);
  // imgui brings its own opengl loader
  // https://github.com/ocornut/imgui/issues/4445
  ;
}
GraphicsFramework::~GraphicsFramework() {
  {

    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
                << (__LINE__) << (" ") << (__func__) << (" ")
                << ("destroy window") << (" ") << (std::endl) << (std::flush);
  }
  glfwDestroyWindow(window.get());
  {

    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
                << (__LINE__) << (" ") << (__func__) << (" ")
                << ("disable GLFW error callback") << (" ") << (std::endl)
                << (std::flush);
  }
  glfwSetErrorCallback(nullptr);
  {

    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
                << (__LINE__) << (" ") << (__func__) << (" ")
                << ("terminate GLFW") << (" ") << (std::endl) << (std::flush);
  }
  glfwTerminate();
}
bool GraphicsFramework::WindowShouldClose() {
  return glfwWindowShouldClose(window.get());
}
void GraphicsFramework::PollEvents() { glfwPollEvents(); }
std::shared_ptr<GLFWwindow> GraphicsFramework::getWindow() { return window; }