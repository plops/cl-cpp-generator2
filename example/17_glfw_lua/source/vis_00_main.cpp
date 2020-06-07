
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
#include <cassert>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
using namespace std::chrono_literals;
State state = {};
void mainLoop() {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("mainLoop") << (" ")
      << (std::endl) << (std::flush);
  while (!(glfwWindowShouldClose(state._window))) {
    glfwPollEvents();
    glfwGetCursorPos(state._window, &(state._cursor_xpos),
                     &(state._cursor_ypos));
    drawFrame();
    drawGui();
    glfwSwapBuffers(state._window);
  }

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("exit mainLoop")
      << (" ") << (std::endl) << (std::flush);
}
void run() {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("start run") << (" ")
      << (std::endl) << (std::flush);
  initWindow();
  initGui();
  initDraw();
  initLua();
  mainLoop();

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("finish run") << (" ")
      << (std::endl) << (std::flush);
};
int main() {
  state._main_version = "76dad8ce8d92733dc99c8c08d9a86a7f07d2e775";
  state._code_repository = "http://10.1.10.5:30080/martin/py_wavelength_tune/";
  state._code_generation_time = "16:00:23 of Sunday, 2020-06-07 (GMT+1)";
  state._start_time =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("start main") << (" ")
      << (std::setw(8)) << (" state._main_version='") << (state._main_version)
      << ("'") << (std::setw(8)) << (" state._code_repository='")
      << (state._code_repository) << ("'") << (std::setw(8))
      << (" state._code_generation_time='") << (state._code_generation_time)
      << ("'") << (std::endl) << (std::flush);
  run();

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("start cleanups")
      << (" ") << (std::endl) << (std::flush);
  cleanupLua();
  cleanupDraw();
  cleanupGui();
  cleanupWindow();

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("end main") << (" ")
      << (std::endl) << (std::flush);
  return 0;
};