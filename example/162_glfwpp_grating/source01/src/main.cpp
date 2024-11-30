#include <format>
#include <glfwpp/glfwpp.h>
#include <iostream>

int main(int argc, char **argv) {
  auto GLFW{glfw::init()};
  auto hints{glfw::WindowHints{.clientApi = glfw::ClientApi::OpenGl,
                               .contextVersionMajor = 4,
                               .contextVersionMinor = 6}};
  hints.apply();
  auto window{glfw::Window(800, 600, "GLFWPP Grating")};
  glfw::makeContextCurrent(window);
  while (!window.shouldClose()) {
    auto time{glfw::getTime()};
    glClearColor(1.0F, 0.20F, 0.30F, 1.0F);
    glClear(GL_COLOR_BUFFER_BIT);
    glfw::pollEvents();
    window.swapBuffers();
  }
  return 0;
}
