#include "MainWindow.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <thread>
std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_opengl3_loader.h"
#include "implot.h"
#include <GLFW/glfw3.h>
// https://gist.github.com/TheOpenDevProject/1662fa2bfd8ef087d94ad4ed27746120

class DestroyGLFWwindow {
public:
  void operator()(GLFWwindow *ptr) {
    // Destroy GLFW window context.

    glfwDestroyWindow(ptr);
    glfwTerminate();
  }
};

int main(int argc, char **argv) {
  (g_start_time) = (std::chrono::high_resolution_clock::now());
  // start

  // glfw initialization

  // https://github.com/ocornut/imgui/blob/docking/examples/example_glfw_opengl3/main.cpp

  glfwSetErrorCallback([&](int err, const char *description) {
    // glfw error
  });
  if (!(glfwInit())) {
    // glfwInit failed.
  }
  const char *glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  std::unique_ptr<GLFWwindow, DestroyGLFWwindow> window;
  auto w{glfwCreateWindow(1280, 720, "dear imgui example", nullptr, nullptr)};
  if ((nullptr) == (w)) {
    // glfwCreatWindow failed.
  }
  window.reset(w);
  glfwMakeContextCurrent(window.get());
  // enable vsync

  glfwSwapInterval(1);
  // imgui brings its own opengl loader
  // https://github.com/ocornut/imgui/issues/4445

  MainWindow M;
  M.Init(window.get(), glsl_version);
  while (!glfwWindowShouldClose(window.get())) {
    glfwPollEvents();
    M.NewFrame();
    M.Update();
    M.Render(window.get());
  }
  // cleanup

  M.Shutdown();
  // leave program

  return 0;
}
