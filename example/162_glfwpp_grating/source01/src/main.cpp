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
  auto window{glfw::Window(800, 600, "GLFWPP Grating")};
  glfw::makeContextCurrent(window);
  while (!window.shouldClose()) {
    auto time{glfw::getTime()};
    glClearColor(0.F, 0.F, 0.F, 1.0F);
    glClear(GL_COLOR_BUFFER_BIT);
    glfw::pollEvents();
    glPushMatrix();
    {
      // Setup modelview matrix (flat XY view)
      mat4x4 view;
      vec3 eye{0.F, 0.F, 1.0F};
      vec3 center{0.F, 0.F, 0.F};
      vec3 up{0.F, 1.0F, 0.F};
      mat4x4_look_at(view, eye, center, up);
      glLoadMatrixf(reinterpret_cast<const GLfloat *>(view));
    }
    glColor4f(1.0F, 1.0F, 1.0F, 1.0F);
    glBegin(GL_LINES);
    glVertex2f(0.F, 0.F);
    glVertex2f(1.0F, 1.0F);
    glEnd();
    glPopMatrix();
    window.swapBuffers();
  }
  return 0;
}
