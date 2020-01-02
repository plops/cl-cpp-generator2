
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;
void keyCallback(GLFWwindow *window, int key, int scancode, int action,
                 int mods) {
  if ((((((key) == (GLFW_KEY_ESCAPE)) || ((key) == (GLFW_KEY_Q)))) &&
       ((action) == (GLFW_PRESS)))) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  };
}
void errorCallback(int err, const char *description) {
  std::setprecision(3);
  (std::cout) << (std::setw(10))
              << (((std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count()) -
                   (state._start_time)))
              << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
              << (__func__) << (" ") << ("error") << (" ") << (std::setw(8))
              << (" err=") << (err) << (std::setw(8)) << (" description=")
              << (description) << (std::endl);
}
static void framebufferResizeCallback(GLFWwindow *window, int width,
                                      int height) {
  std::setprecision(3);
  (std::cout) << (std::setw(10))
              << (((std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count()) -
                   (state._start_time)))
              << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
              << (__func__) << (" ") << ("resize") << (" ") << (std::setw(8))
              << (" width=") << (width) << (std::setw(8)) << (" height=")
              << (height) << (std::endl);
  auto app = (State *)(glfwGetWindowUserPointer(window));
  app->_framebufferResized = true;
}
void initWindow() {
  if (glfwInit()) {
    glfwSetErrorCallback(errorCallback);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    state._window = glfwCreateWindow(800, 600, "doppler window", NULL, NULL);
    glfwSetKeyCallback(state._window, keyCallback);
    glfwSetWindowUserPointer(state._window, &(state));
    glfwSetFramebufferSizeCallback(state._window, framebufferResizeCallback);
    glfwMakeContextCurrent(state._window);
  };
}
void cleanupWindow() {
  glfwDestroyWindow(state._window);
  glfwTerminate();
};