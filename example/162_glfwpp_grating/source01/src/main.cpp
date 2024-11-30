#include <format>
#include <glfwpp/glfwpp.h>
#include <iostream>

int main(int argc, char **argv) {
  std::cout << std::format("(:thread::hardware_concurrency() '{}')\n",
                           thread::hardware_concurrency());
  auto GLFW{glfw::init()};
  auto hints{
      glfw::WindowHints({clientApi, glfw::ClientApi::OpenGl,
                         contextVersionMajor, 4, contextVersionMinor, 6})};
  hints.apply();
  auto window{glfw::Window(800, 600, "GLFWPP Grating")};
  glfw::makeContextCurrent(window);
  if (GLEW_OK != glewInit()) {
    throw std::runtime_error("Could not initialize GLEW");
  }
  while (!window.shouldClose()) {
    auto time{glfw::getTime()};
    glClearColor(1.0F, 0.20F, 0.30F, 1.0F);
    glCear(GL_COLOR_BUFFER_BIT);
    glfw::pollEvent();
    window.swapBuffers();
  }
  return 0;
}
