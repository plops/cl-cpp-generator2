// no preamble
#include <chrono>
#include <iostream>
#include <spdlog/spdlog.h>
#include <thread>
extern const std::chrono::time_point<std::chrono::high_resolution_clock>
    g_start_time;
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <imgui.h>
#define GLFW_INCLUDE_NONE
#include "GlfwWindow.h"
#include <GLFW/glfw3.h>
GlfwWindow::GlfwWindow() {
  spdlog::info("initialize GLFW3");
  if (!(glfwInit())) {
    spdlog::info("glfwInit failed");
  }
  glfwWindowHint(GLFW_VISIBLE, true);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, true);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  spdlog::info("create GLFW3 window");
  const auto startWidth{800};
  const auto startHeight{600};
  auto window{
      glfwCreateWindow(startWidth, startHeight, "glfw", nullptr, nullptr)};
  if (!(window)) {
    spdlog::info("can't create glfw window");
  }
  spdlog::info("initialize GLFW3 context for window");
  glfwMakeContextCurrent(window);
  // configure Vsync, 1 locks to 60Hz, FIXME: i should really check glfw errors
  glfwSwapInterval(0);
  (m_window) = (window);
}
GlfwWindow::~GlfwWindow() {
  spdlog::info("");
  glfwDestroyWindow(m_window);
  glfwTerminate();
}
bool GlfwWindow::WindowShouldClose() { return glfwWindowShouldClose(m_window); }
void GlfwWindow::SwapBuffers() { glfwSwapBuffers(m_window); }
GLFWwindow *GlfwWindow::GetWindow() { return m_window; }
void GlfwWindow::PollEvents() { glfwPollEvents(); }
GLFWglproc GlfwWindow::GetProcAddress(const char *name) {
  spdlog::info("  name='{}'", name);
  return glfwGetProcAddress(name);
}
std::pair<int, int> GlfwWindow::GetWindowSize() const {
  auto width{int(0)};
  auto height{int(0)};
  glfwGetWindowSize(m_window, &width, &height);
  return std::make_pair(width, height);
}