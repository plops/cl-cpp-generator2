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
struct uchar4;
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
enum { TX = 32, TY = 32, RAD = 1 };
int divUp(int a, int b) { return ((((a) + (b) + (-1))) / (b)); }
__device__ unsigned char clip(int n) {
  if (255 < n) {
    return 255;
  } else {
    if (n < 0) {
      return 0;
    } else {
      return n;
    }
  }
}
__device__ int idxClip(int n, int ma) {
  if (((ma) - (1)) < n) {
    return ((ma) - (1));
  } else {
    if (n < 0) {
      return 0;
    } else {
      return n;
    }
  }
}
__device__ int flatten(int col, int row, int w, int h) {
  return ((idxClip(col, w)) + (((w) * (idxClip(row, h)))));
};
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
__global__ void tempKernel(uchar4 *d_out, float *d_temp, int w, int h, BC bc) {
  extern __shared__ float s_in[];
  auto col = ((((blockIdx.x) * (blockDim.x))) + (threadIdx.x));
  auto row = ((((blockIdx.y) * (blockDim.y))) + (threadIdx.y));
  if ((((w) <= (col)) || ((h) <= (row)))) {
    return;
  };
  auto idx = flatten(col, row, w, h);
  auto s_w = ((blockDim.x) + (((2) * (RAD))));
  auto s_h = ((blockDim.y) + (((2) * (RAD))));
  auto s_col = ((threadIdx.x) + (RAD));
  auto s_row = ((threadIdx.y) + (RAD));
  auto s_idx = flatten(s_col, s_row, s_w, s_h);
  d_out[idx].x = 0;
  d_out[idx].y = 0;
  d_out[idx].z = 0;
  d_out[idx].w = 255;
  s_in[s_idx] = d_temp[idx];
  if (threadIdx.x < RAD) {
    s_in[flatten(((s_col) - (RAD)), s_row, s_w, s_h)] =
        d_temp[flatten(((col) - (RAD)), row, w, h)];
    s_in[flatten(((s_col) + (RAD)), s_row, s_w, s_h)] =
        d_temp[flatten(((col) + (RAD)), row, w, h)];
  };
  if (threadIdx.y < RAD) {
    s_in[flatten(s_col, ((s_row) - (RAD)), s_w, s_h)] =
        d_temp[flatten(col, ((row) - (RAD)), w, h)];
    s_in[flatten(s_col, ((s_row) + (blockDim.y)), s_w, s_h)] =
        d_temp[flatten(col, ((row) + (blockDim.y)), w, h)];
  };
}
void kernelLauncher(uchar4 *d_out, float *d_temp, int w, int h, BC bc) {
  const dim3 blockSize(TX, TY);
  const dim3 gridSize(divUp(w, TX), divUp(h, TY));
  auto smSz = ((sizeof(float)) * (((TX) + (((2) * (RAD))))) *
               (((TY) + (((2) * (RAD))))));
  tempKernel<<<gridSize, blockSize, smSz>>>(d_out, d_temp, w, h, bc);
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
    auto d_out = static_cast<uchar4 *>(0);
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
      kernelLauncher(d_out, d_temp, width, height, bc);
      glfwSwapBuffers(window);
    }
    glfwDestroyWindow(window);
  };
  glfwTerminate();
}