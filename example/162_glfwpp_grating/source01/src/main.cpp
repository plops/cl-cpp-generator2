#include "/home/martin/src/glfw/deps/linmath.h"
#include <chrono>
#include <format>
#include <glfwpp/glfwpp.h>
#include <iostream>
#include <thread>

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
  glfw::swapInterval(2);
  while (!window.shouldClose()) {
    auto time{glfw::getTime()};
    glfw::pollEvents();
    // show a sequence of horizontal bars and vertical bars that split the image
    // into 1/2, 1/4th, ... . each image is followed by its inverted version.
    // the lcd of the projector is too slow to show this pattern exactly with
    // 60Hz. that is why we set swap interval to 2 (we wait for two frames for
    // every image so that the display has time to settle)
    static int current_level = 0;
    static bool horizontal = true;
    current_level = current_level + 1;
    if (current_level == 8 * 2) {
      horizontal = !horizontal;
      current_level = 0;
    }
    auto white{0 == (current_level % 2)};
    if (white) {
      glClearColor(0.F, 0.F, 0.F, 1.0F);
      glClear(GL_COLOR_BUFFER_BIT);
      glColor4f(1.0F, 1.0F, 1.0F, 1.0F);
    } else {
      glClearColor(1.0F, 1.0F, 1.0F, 1.0F);
      glClear(GL_COLOR_BUFFER_BIT);
      glColor4f(0.F, 0.F, 0.F, 1.0F);
    }
    glPushMatrix();
    glTranslatef(-1.0F, -1.0F, 0.F);
    glScalef(2.0F / w, 2.0F / h, 1.0F);
    glBegin(GL_QUADS);
    auto level{current_level / 2};
    auto y{1024 / pow(2.0F, level)};
    if (horizontal) {
      for (decltype(0 + pow(2.0F, level) + 1) i = 0; i < pow(2.0F, level);
           i += 1) {
        auto x{512};
        auto o{2 * i * y};
        glVertex2f(0, o);
        glVertex2f(0, o + y);
        glVertex2f(x, o + y);
        glVertex2f(x, o);
      }
    } else {
      for (decltype(0 + pow(2.0F, level) + 1) i = 0; i < pow(2.0F, level);
           i += 1) {
        auto x{512};
        auto o{2 * i * y};
        glVertex2f(o, 0);
        glVertex2f(o + y, 0);
        glVertex2f(o + y, x);
        glVertex2f(o, x);
      }
    }
    glEnd();
    glPopMatrix();
    window.swapBuffers();
  }
  return 0;
}
