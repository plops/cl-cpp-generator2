#ifndef GLOBALS_H

#define GLOBALS_H

#include <GLFW/glfw3.h>

#include <complex>

#include <cmath>

#include <chrono>
struct State {
  typeof(std::chrono::high_resolution_clock::now().time_since_epoch().count())
      _start_time;
  GLuint _fontTex;
  bool _framebufferResized;
  GLFWwindow *_window;
  size_t _mmap_filesize;
  void *_mmap_data;
  std::complex<float> *_range_line;
  int _echo;
  int _range;
  char const *_filename;
};
typedef struct State State;

#endif
