#include "/home/martin/src/glfw/deps/linmath.h"
#include <format>
#include <glfwpp/glfwpp.h>
#include <iostream>

int main(int argc, char **argv) {
  auto GLFW{glfw::init()};
  auto hints{glfw::WindowHints{.clientApi = glfw::ClientApi::OpenGl,
                               .contextVersionMajor = 2,
                               .contextVersionMinor = 0}};
  hints.apply();
  auto w{800};
  auto h{600};
  auto window{glfw::Window(w, h, "GLFWPP Grating")};
  glfw::makeContextCurrent(window);
  while (!window.shouldClose()) {
    auto time{glfw::getTime()};
    glClearColor(0.F, 0.F, 0.F, 1.0F);
    glClear(GL_COLOR_BUFFER_BIT);
    glfw::pollEvents();
    glColor4f(1.0F, 1.0F, 1.0F, 1.0F);
    glBegin(GL_LINES);
    for (decltype(0 + (w / 2) + 1) i = 0; i < (w / 2); i += 1) {
      auto x{(i / (0.250F * w)) - 1};
      glVertex2f(x, -1.0F);
      glVertex2f(x, 1.0F);
    }
    glEnd();
    window.swapBuffers();
  }
  return 0;
}
