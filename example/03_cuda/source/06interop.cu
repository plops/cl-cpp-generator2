// nvcc -o 06interop 06interop.cu -lglfw -lGL -march=native --std=c++14 -O3 -g
// note that nvcc requires gcc 8
// nvprof 06interop
#include <GLFW/glfw3.h>
#include <cassert>
#include <cstdio>
#include <iostream>
void key_callback(GLFWwindow *window, int key, int scancode, int action,
                  int mods) {
  if ((((((key) == (GLFW_KEY_ESCAPE)) || ((key) == (GLFW_KEY_Q)))) &&
       ((action) == (GLFW_PRESS)))) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  };
}
void error_callback(int err, const char *description) {
  fprintf(stderr, "Error: %s\n", description);
}
using namespace std;
int main() {
  (cout) << ("bla") << (endl);
  if (glfwInit()) {
    glfwSetErrorCallback(error_callback);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    auto window = glfwCreateWindow(640, 480, "cuda interop", NULL, NULL);
    assert(window);
    glfwSetKeyCallback(window, key_callback);
    glfwMakeContextCurrent(window);
    int width;
    int height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);
    glfwSwapInterval(1);
    glClearColor(0, 0, 0, 0);
    while (!(glfwWindowShouldClose(window))) {
      glfwPollEvents();
      auto time = glfwGetTime();
      glClear(GL_COLOR_BUFFER_BIT);
      glfwSwapBuffers(window);
    }
    glfwDestroyWindow(window);
  };
  glfwTerminate();
}