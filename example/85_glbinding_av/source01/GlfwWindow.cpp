// no preamble
#include <chrono>
#include <iostream>
#include <thread>
void lprint(std::initializer_list<std::string> il, std::string file, int line,
            std::string fun);
extern std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <imgui.h>
#define GLFW_INCLUDE_NONE
#include "GlfwWindow.h"
#include <GLFW/glfw3.h>
GlfwWindow::GlfwWindow() {
  lprint({"initialize GLFW3", " "}, __FILE__, __LINE__,
         &(__PRETTY_FUNCTION__[0]));
  if (!(glfwInit())) {
    lprint({"glfwInit failed", " "}, __FILE__, __LINE__,
           &(__PRETTY_FUNCTION__[0]));
  }
  glfwWindowHint(GLFW_VISIBLE, true);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, true);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  lprint({"create GLFW3 window", " "}, __FILE__, __LINE__,
         &(__PRETTY_FUNCTION__[0]));
  const auto startWidth;
  const auto startHeight = 600;
  auto window =
      glfwCreateWindow(startWidth, startHeight, "glfw", nullptr, nullptr);
  if (!(window)) {
    lprint({"can't create glfw window", " "}, __FILE__, __LINE__,
           &(__PRETTY_FUNCTION__[0]));
  }
  lprint({"initialize GLFW3 context for window", " "}, __FILE__, __LINE__,
         &(__PRETTY_FUNCTION__[0]));
  glfwMakeContextCurrent(window);
  // configure Vsync, 1 locks to 60Hz, FIXME: i should really check glfw errors
  glfwSwapInterval(0);
  m_window = window;
}
GlfwWindow::~GlfwWindow() {
  lprint({"", " "}, __FILE__, __LINE__, &(__PRETTY_FUNCTION__[0]));
  glfwDestroyWindow(m_window);
  glfwTerminate();
}
bool GlfwWindow::WindowShouldClose() { return glfwWindowShouldClose(m_window); }
void GlfwWindow::SwapBuffers() { glfwSwapBuffers(m_window); }
GLFWwindow *GlfwWindow::GetWindow() { return m_window; }
std::pair<int, int> GlfwWindow::GetWindowSize() {
  auto width = int(0);
  auto height = int(0);
  glfwGetWindowSize(m_window, &width, &height);
  return std::make_pair(width, height);
}