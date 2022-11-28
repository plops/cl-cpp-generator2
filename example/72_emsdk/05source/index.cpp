// no preamble
;
#include "index.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
std::mutex g_stdout_mutex;
void reset_state() {
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);
  glDisable(GL_CULL_FACE);
  glDisable(GL_TEXTURE_2D);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glMatrixMode(GL_TEXTURE);
  glLoadIdentity();
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
}
void draw_triangle() {
  reset_state();
  glBegin(GL_TRIANGLES);
  glColor3f((1.0f), (0.f), (0.f));
  glVertex2f((0.f), (0.50f));
  glColor3f((0.f), (0.f), (1.0f));
  glVertex2f((-0.50f), (-0.50f));
  glColor3f((0.f), (1.0f), (0.f));
  glVertex2f((0.50f), (-0.50f));
  glEnd();
}
std::function<void()> loop;
void main_loop() { loop(); }
int main(int argc, char **argv) {
  glfwInit();
  glfwWindowHint(GLFW_SAMPLES, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 1);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  auto *w = glfwCreateWindow(512, 512, "test glfw", nullptr, nullptr);
  glfwMakeContextCurrent(w);
  glfwSwapInterval(1);
  flextInit(w);
  loop = [&]() {
    auto dw = 0;
    auto dh = 0;
    glfwGetFramebufferSize(w, &dw, &dh);
    auto ww = ((dh) / (2));
    auto hh = ((dh) / (2));
    auto x0 = ((((dw) / (2))) - (hh));
    auto x1 = ((dw) / (2));
    auto y0 = ((dh) / (2));
    auto y1 = 0;
    glClearColor((0.f), (0.f), (0.f), (1.0f));
    glClearDepth((1.0f));
    glClear(((GL_COLOR_BUFFER_BIT) | (GL_DEPTH_BUFFER_BIT)));
    glViewport(x0, y0, ww, hh);
    draw_triangle();
    glfwSwapBuffers(w);
    glfwPollEvents();
  };
#ifdef __EMSCRIPTEN__
  emscripten_set_main_loop(main_loop, 0, true);
#else
  while (!glfwWindowShouldClose(w)) {
    main_loop();
  }
#endif
  glfwDestroyWindow(w);
  glfwTerminate();
  return 0;
}