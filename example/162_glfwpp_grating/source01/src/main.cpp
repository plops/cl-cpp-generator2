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
  auto w{512};
  auto h{512};
  auto window{glfw::Window(w, h, "GLFWPP Grating")};
  glfw::makeContextCurrent(window);
  glfw::swapInterval(1);
  while (!window.shouldClose()) {
    auto time{glfw::getTime()};
    glClearColor(0.F, 0.F, 0.F, 1.0F);
    glClear(GL_COLOR_BUFFER_BIT);
    glfw::pollEvents();
    glColor4f(1.0F, 1.0F, 1.0F, 1.0F);
    glBegin(GL_LINES);
    auto skip{32};
    static int offset = 0;
    offset = ((offset + 1) % skip);
    auto N{h / skip};
    auto Nx{w / skip};
    for (decltype(0 + Nx + 2 + 1) i = 0; i < Nx + 2; i += 1) {
      auto x{(i - (Nx / 2) - (offset / (1.0F * N))) / ((0.50F / skip) * w)};
      glVertex2f(x, -1.0F);
      glVertex2f(x, 1.0F);
    }
    glEnd();
    window.swapBuffers();
  }
  return 0;
}
