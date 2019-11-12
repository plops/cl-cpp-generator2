// nvcc -o 06interop 06interop.cu -lglfw -lGL -march=native --std=c++14 -O3 -g
// note that nvcc requires gcc 8
// nvprof 06interop
#include <GLFW/glfw3.h>
#include <cassert>
#include <cstdio>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
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
struct BC {
  int x;
  int y;
  float rad;
  int chamfer;
  float t_s;
  float t_a;
  float t_g;
};
typedef struct BC BC;
enum { TX = 32, TY = 32 };
int divUp(int a, int b) { return ((((a) + (b) + (-1))) / (b)); }
__global__ void resetKernel(float *d_temp, int w, int h, BC bc) {
  auto col = ((((blockIdx.x) * (blockDim.x))) + (threadIdx.x));
  auto row = ((((blockIdx.y) * (blockDim.y))) + (threadIdx.y));
  if ((((w) <= (col)) || ((h) <= (row)))) {
    return;
  };
  d_temp[((col) + (((row) * (w))))] = bc.t_a;
}
void resetTemperature(float *d_temp, int w, int h, BC bc) {
  const dim3 blockSize(TX, TY);
  const dim3 gridSize(divUp(w, TX), divUp(h, TY));
  resetKernel<<<gridSize, blockSize>>>(d_temp, w, h, bc);
}
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
    auto d_temp = static_cast<float *>(0);
    auto bc = (BC){((width) / (2)),
                   ((height) / (2)),
                   ((width) / ((1.e+1f))),
                   150,
                   (2.12e+2f),
                   (7.e+1f),
                   (0.0e+0f)};
    cudaMalloc(&d_temp, ((width) * (height) * (sizeof(*d_temp))));
    resetTemperature(d_temp, width, height, bc);
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