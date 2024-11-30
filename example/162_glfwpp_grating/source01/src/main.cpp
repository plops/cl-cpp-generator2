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
    // tree
    glPushMatrix();
    glTranslatef(-1.0F, -1.0F, 0.F);
    glScalef(2.0F / w, 2.0F / h, 1.0F);
    glBegin(GL_QUADS);
    auto level{3};
    for (decltype(0 + level + 1) i = 0; i < level; i += 1) {
      auto x{512};
      auto y{1024 / std::pow(2.0F, level)};
      auto o{2 * i * y};
      glColor4f((1.0F * i) / level, 1.0F, 1.0F, 1.0F);
      glVertex2f(0, o);
      glVertex2f(0, o + y);
      glVertex2f(x, o + y);
      glVertex2f(x, o);
    }
    glEnd();
    glPopMatrix();
    window.swapBuffers();
  }
  return 0;
}
