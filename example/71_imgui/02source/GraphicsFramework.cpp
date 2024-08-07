// no preamble

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
  glfwSetErrorCallback([&](int err, const char *description) {
    // glfw error
  });
  // glfw initialization

  // https://github.com/ocornut/imgui/blob/docking/examples/example_glfw_opengl3/main.cpp

  if (!(glfwInit())) {
    // glfwInit failed.
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  auto w{glfwCreateWindow(1800, 1000, "dear imgui example", nullptr, nullptr)};
  if ((nullptr) == (w)) {
    // glfwCreatWindow failed.
  }
  window.reset(w, [&](GLFWwindow *ptr) { glfwDestroyWindow(ptr); });
  glfwMakeContextCurrent(window.get());
  // enable vsync

  glfwSwapInterval(1);
  // imgui brings its own opengl loader
  // https://github.com/ocornut/imgui/issues/4445
}
GraphicsFramework::~GraphicsFramework() {
  // destroy window

  glfwDestroyWindow(window.get());
  // disable GLFW error callback

  glfwSetErrorCallback(nullptr);
  // terminate GLFW

  glfwTerminate();
}
bool GraphicsFramework::WindowShouldClose() {
  return glfwWindowShouldClose(window.get());
}
void GraphicsFramework::PollEvents() { glfwPollEvents(); }
std::shared_ptr<GLFWwindow> GraphicsFramework::getWindow() { return window; }