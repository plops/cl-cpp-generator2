#pragma once
#include "implot.h"
#include <condition_variable>
#include <deque>
#include <mutex>
// https://gist.github.com/TheOpenDevProject/1662fa2bfd8ef087d94ad4ed27746120
;
class DestroyGLFWwindow {
public:
  void operator()(GLFWwindow *ptr) {
    {

      std::chrono::duration<double> timestamp =
          std::chrono::high_resolution_clock::now() - g_start_time;
      (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                  << (std::this_thread::get_id()) << (" ") << (__FILE__)
                  << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                  << ("Destroy GLFW window context.") << (" ") << (std::endl)
                  << (std::flush);
    }
    glfwDestroyWindow(ptr);
    glfwTerminate();
  }
};