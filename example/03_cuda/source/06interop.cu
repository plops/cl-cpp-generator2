// glad --generator=c-debug --spec=gl --out-path=GL
// --extensions=GL_EXT_framebuffer_multisample,GL_EXT_texture_filter_anisotropic
// nvcc -o 06interop GL/src/glad.c 06interop.cu -IGL/include -lglfw -lGL
// --std=c++14 -O3 -g -Xcompiler=-march=native -Xcompiler=-ggdb -ldl note that
// nvcc requires gcc 8 nvprof 06interop
// https://github.com/myurtoglu/cudaforengineers/tree/master/heat_2d
// 2019 duane storti cuda for engineers p. 90
#include <glad/glad.h>

#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

#include <cuda_gl_interop.h>
struct uchar4;
struct BC {
  int x;
  int y;
  float rad;
  int chamfer;
  float t_s;
  float t_a;
  float t_g;
  float *d_temp;
  int width;
  int height;
};
typedef struct BC BC;
enum { TX = 16, TY = 16, RAD = 1, ITERS_PER_RENDER = 1000 };
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
auto g_start = static_cast<typeof(
    std::chrono::high_resolution_clock::now().time_since_epoch().count())>(0);
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
void key_callback(GLFWwindow *window, int key, int scancode, int action,
                  int mods) {
  auto bc = static_cast<BC *>(glfwGetWindowUserPointer(window));
  auto DT = (1.e+0f);
  if ((((((key) == (GLFW_KEY_ESCAPE)) || ((key) == (GLFW_KEY_Q)))) &&
       ((action) == (GLFW_PRESS)))) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (g_start)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__)
                << (" glfwSetWindowShouldClose(window, GLFW_TRUE) ")
                << (std::endl);
  };
  if ((((((key) == (GLFW_KEY_M)))) && ((action) == (GLFW_PRESS)))) {
    resetTemperature(bc->d_temp, bc->width, bc->height, *bc);
    (std::cout)
        << (((std::chrono::high_resolution_clock::now()
                  .time_since_epoch()
                  .count()) -
             (g_start)))
        << (" ") << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
        << (" resetTemperature(bc->d_temp, bc->width, bc->height, *bc) ")
        << (std::endl);
  };
  if ((((key) == (GLFW_KEY_1)) &&
       ((((action) == (GLFW_PRESS)) || ((action) == (GLFW_REPEAT)))))) {
    (bc->t_s) += (DT);
  };
  if ((((key) == (GLFW_KEY_2)) &&
       ((((action) == (GLFW_PRESS)) || ((action) == (GLFW_REPEAT)))))) {
    (bc->t_s) -= (DT);
  };
  if ((((key) == (GLFW_KEY_3)) &&
       ((((action) == (GLFW_PRESS)) || ((action) == (GLFW_REPEAT)))))) {
    (bc->t_a) += (DT);
  };
  if ((((key) == (GLFW_KEY_4)) &&
       ((((action) == (GLFW_PRESS)) || ((action) == (GLFW_REPEAT)))))) {
    (bc->t_a) -= (DT);
  };
  if ((((key) == (GLFW_KEY_5)) &&
       ((((action) == (GLFW_PRESS)) || ((action) == (GLFW_REPEAT)))))) {
    (bc->t_g) += (DT);
  };
  if ((((key) == (GLFW_KEY_6)) &&
       ((((action) == (GLFW_PRESS)) || ((action) == (GLFW_REPEAT)))))) {
    (bc->t_g) -= (DT);
  };
  if ((((key) == (GLFW_KEY_7)) &&
       ((((action) == (GLFW_PRESS)) || ((action) == (GLFW_REPEAT)))))) {
    (bc->chamfer) += (1);
  };
  if ((((key) == (GLFW_KEY_8)) &&
       ((((action) == (GLFW_PRESS)) || ((action) == (GLFW_REPEAT)))))) {
    (bc->chamfer) -= (1);
  };
  if ((((key) == (GLFW_KEY_9)) &&
       ((((action) == (GLFW_PRESS)) || ((action) == (GLFW_REPEAT)))))) {
    (bc->rad) += ((2.e+0f));
  };
  if ((((key) == (GLFW_KEY_0)) &&
       ((((action) == (GLFW_PRESS)) || ((action) == (GLFW_REPEAT)))))) {
    (bc->rad) -= ((2.e+0f));
  };
  char s[1024];
  snprintf(s, 1023,
           "cuda pipe=%.2g air=%.2g ground=%.2g chamfer=%d radius=%.2g",
           bc->t_s, bc->t_a, bc->t_g, bc->chamfer, bc->rad);
  glfwSetWindowTitle(window, s);
}
void error_callback(int err, const char *description) {
  fprintf(stderr, "Error: %s\n", description);
}
using namespace std;
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
  d_out[idx].y = 128;
  d_out[idx].z = 0;
  d_out[idx].w = 255;
  s_in[s_idx] = d_temp[idx];
  if (threadIdx.x < RAD) {
    s_in[flatten(((s_col) - (RAD)), s_row, s_w, s_h)] =
        d_temp[flatten(((col) - (RAD)), row, w, h)];
    s_in[flatten(((s_col) + (blockDim.x)), s_row, s_w, s_h)] =
        d_temp[flatten(((col) + (blockDim.x)), row, w, h)];
  };
  if (threadIdx.y < RAD) {
    s_in[flatten(s_col, ((s_row) - (RAD)), s_w, s_h)] =
        d_temp[flatten(col, ((row) - (RAD)), w, h)];
    s_in[flatten(s_col, ((s_row) + (blockDim.y)), s_w, s_h)] =
        d_temp[flatten(col, ((row) + (blockDim.y)), w, h)];
  };
  auto dSq = ((((((col) - (bc.x))) * (((col) - (bc.x))))) +
              (((((row) - (bc.y))) * (((row) - (bc.y))))));
  if (dSq < ((bc.rad) * (bc.rad))) {
    d_temp[idx] = bc.t_s;
    return;
  };
  if ((((0) == (col)) || ((((w) - (1))) == (col)) || ((0) == (row)) ||
       (((col) + (row)) < bc.chamfer))) {
    d_temp[idx] = bc.t_a;
    return;
  };
  if ((((h) - (1))) == (row)) {
    d_temp[idx] = bc.t_g;
    return;
  };
  __syncthreads();
  auto temp =
      (((2.5e-1f)) * (((s_in[flatten(((s_col) - (1)), s_row, s_w, s_h)]) +
                       (s_in[flatten(((s_col) + (1)), s_row, s_w, s_h)]) +
                       (s_in[flatten(s_col, ((s_row) - (1)), s_w, s_h)]) +
                       (s_in[flatten(s_col, ((s_row) + (1)), s_w, s_h)]))));
  d_temp[idx] = temp;
  auto intensity = clip((int)temp);
  d_out[idx].x = intensity;
  d_out[idx].z = ((255) - (intensity));
}
void kernelLauncher(uchar4 *d_out, float *d_temp, int w, int h, BC bc) {
  const dim3 blockSize(TX, TY);
  const dim3 gridSize(divUp(w, TX), divUp(h, TY));
  auto smSz = ((sizeof(float)) * (((TX) + (((2) * (RAD))))) *
               (((TY) + (((2) * (RAD))))));
  tempKernel<<<gridSize, blockSize, smSz>>>(d_out, d_temp, w, h, bc);
}
auto g_cuda_pbo_resource = static_cast<struct cudaGraphicsResource *>(0);
void render(float *d_temp, int w, int h, BC bc) {
  {
    auto r = cudaGraphicsMapResources(1, &g_cuda_pbo_resource, 0);
    if (!((cudaSuccess) == (r))) {
      (std::cout)
          << (((std::chrono::high_resolution_clock::now()
                    .time_since_epoch()
                    .count()) -
               (g_start)))
          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
          << (" cudaGraphicsMapResources(1, &g_cuda_pbo_resource, 0) => ")
          << (r) << (" '") << (cudaGetErrorString(r)) << ("' ")
          << (" g_cuda_pbo_resource=") << (g_cuda_pbo_resource) << (std::endl);
    };
    assert((cudaSuccess) == (r));
  };
  auto d_out = static_cast<uchar4 *>(0);
  {
    auto r = cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void **>(&d_out), nullptr, g_cuda_pbo_resource);
    if (!((cudaSuccess) == (r))) {
      (std::cout) << (((std::chrono::high_resolution_clock::now()
                            .time_since_epoch()
                            .count()) -
                       (g_start)))
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__)
                  << (" cudaGraphicsResourceGetMappedPointer(reinterpret_cast<"
                      "void**>(&d_out), nullptr, g_cuda_pbo_resource) => ")
                  << (r) << (" '") << (cudaGetErrorString(r)) << ("' ")
                  << (std::endl);
    };
    assert((cudaSuccess) == (r));
  };
  for (int i = 0; i < ITERS_PER_RENDER; (i) += (1)) {
    kernelLauncher(d_out, d_temp, w, h, bc);
  }
  {
    auto r = cudaGraphicsUnmapResources(1, &g_cuda_pbo_resource, 0);
    if (!((cudaSuccess) == (r))) {
      (std::cout)
          << (((std::chrono::high_resolution_clock::now()
                    .time_since_epoch()
                    .count()) -
               (g_start)))
          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
          << (" cudaGraphicsUnmapResources(1, &g_cuda_pbo_resource, 0) => ")
          << (r) << (" '") << (cudaGetErrorString(r)) << ("' ") << (std::endl);
    };
    assert((cudaSuccess) == (r));
  };
};
void draw_texture(int w, int h) {
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
               nullptr);
  glEnable(GL_TEXTURE_2D);
  glBegin(GL_QUADS);
  glTexCoord2f(0, 0);
  glVertex2f((-1.e+0f), (-1.e+0f));
  glTexCoord2f(0, 1);
  glVertex2f((-1.e+0f), (1.e+0f));
  glTexCoord2f(1, 1);
  glVertex2f((1.e+0f), (1.e+0f));
  glTexCoord2f(1, 0);
  glVertex2f((1.e+0f), (-1.e+0f));
  glEnd();
  glDisable(GL_TEXTURE_2D);
}
int main() {
  g_start =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  auto n_cuda = 0;
  {
    auto r = cudaGetDeviceCount(&n_cuda);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (g_start)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" cudaGetDeviceCount(&n_cuda) => ") << (r)
                << (" '") << (cudaGetErrorString(r)) << ("' ") << (" n_cuda=")
                << (n_cuda) << (std::endl);
    assert((cudaSuccess) == (r));
  };
  {
    auto r = cudaSetDevice(0);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (g_start)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" cudaSetDevice(0) => ") << (r) << (" '")
                << (cudaGetErrorString(r)) << ("' ") << (std::endl);
    assert((cudaSuccess) == (r));
  };
  if (glfwInit()) {
    glfwSetErrorCallback(error_callback);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (g_start)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" glfwSetErrorCallback(error_callback) ")
                << (std::endl);
    auto window = glfwCreateWindow(640, 480, "cuda interop", NULL, NULL);
    assert(window);
    glfwSetKeyCallback(window, key_callback);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (g_start)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" glfwSetKeyCallback(window, key_callback) ")
                << (" window=") << (window) << (std::endl);
    glfwMakeContextCurrent(window);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (g_start)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" glfwMakeContextCurrent(window) ")
                << (std::endl);
    assert(gladLoadGL());
    (cout) << ("GL version ") << (GLVersion.major) << (" ") << (GLVersion.minor)
           << (endl);
    gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress));
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (g_start)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__)
                << (" gladLoadGLLoader(reinterpret_cast<GLADloadproc>("
                    "glfwGetProcAddress)) ")
                << (std::endl);
    int width;
    int height;
    glfwGetFramebufferSize(window, &width, &height);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (g_start)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__)
                << (" glfwGetFramebufferSize(window, &width, &height) ")
                << (" width=") << (width) << (" height=") << (height)
                << (std::endl);
    glViewport(0, 0, width, height);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (g_start)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" glViewport(0, 0, width, height) ")
                << (std::endl);
    glfwSwapInterval(1);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (g_start)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" glfwSwapInterval(1) ") << (std::endl);
    glClearColor(0, 0, 0, 0);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (g_start)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" glClearColor(0, 0, 0, 0) ") << (std::endl);
    auto d_temp = static_cast<float *>(0);
    {
      auto r = cudaMalloc(&d_temp, ((width) * (height) * (sizeof(float))));
      (std::cout)
          << (((std::chrono::high_resolution_clock::now()
                    .time_since_epoch()
                    .count()) -
               (g_start)))
          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
          << (" cudaMalloc(&d_temp, ((width)*(height)*(sizeof(float)))) => ")
          << (r) << (" '") << (cudaGetErrorString(r)) << ("' ") << (" width=")
          << (width) << (" height=") << (height)
          << (" ((((width)*(height)*(sizeof(float))))/"
              "(((1024)*((1.024e+3f)))))=")
          << (((((width) * (height) * (sizeof(float)))) /
               (((1024) * ((1.024e+3f))))))
          << (std::endl);
      assert((cudaSuccess) == (r));
    };
    auto bc = (BC){((width) / (2)),
                   ((height) / (2)),
                   ((width) / ((1.e+1f))),
                   150,
                   (2.12e+2f),
                   (7.e+1f),
                   (0.0e+0f),
                   d_temp,
                   width,
                   height};
    glfwSetWindowUserPointer(window, static_cast<void *>(&bc));
    resetTemperature(d_temp, width, height, bc);
    GLuint pbo = 0;
    GLuint tex = 0;
    glGenBuffers(1, &pbo);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (g_start)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" glGenBuffers(1, &pbo) ") << (" pbo=")
                << (pbo) << (std::endl);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (g_start)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo) ")
                << (" pbo=") << (pbo) << (std::endl);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,
                 ((width) * (height) * (sizeof(GLubyte)) * (4)), 0,
                 GL_STREAM_DRAW);
    (std::cout)
        << (((std::chrono::high_resolution_clock::now()
                  .time_since_epoch()
                  .count()) -
             (g_start)))
        << (" ") << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
        << (" glBufferData(GL_PIXEL_UNPACK_BUFFER, "
            "((width)*(height)*(sizeof(GLubyte))*(4)), 0, GL_STREAM_DRAW) ")
        << (" width=") << (width) << (" height=") << (height)
        << (" ((((width)*(height)*(sizeof(GLubyte))*(4)))/"
            "(((1024)*((1.024e+3f)))))=")
        << (((((width) * (height) * (sizeof(GLubyte)) * (4))) /
             (((1024) * ((1.024e+3f))))))
        << (std::endl);
    glGenTextures(1, &tex);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (g_start)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" glGenTextures(1, &tex) ") << (" tex=")
                << (tex) << (std::endl);
    glBindTexture(GL_TEXTURE_2D, tex);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (g_start)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" glBindTexture(GL_TEXTURE_2D, tex) ")
                << (std::endl);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (g_start)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__)
                << (" glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, "
                    "GL_NEAREST) ")
                << (std::endl);
    {
      auto r = cudaGraphicsGLRegisterBuffer(&g_cuda_pbo_resource, pbo,
                                            cudaGraphicsMapFlagsWriteDiscard);
      (std::cout) << (((std::chrono::high_resolution_clock::now()
                            .time_since_epoch()
                            .count()) -
                       (g_start)))
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__)
                  << (" cudaGraphicsGLRegisterBuffer(&g_cuda_pbo_resource, "
                      "pbo, cudaGraphicsMapFlagsWriteDiscard) => ")
                  << (r) << (" '") << (cudaGetErrorString(r)) << ("' ")
                  << (" g_cuda_pbo_resource=") << (g_cuda_pbo_resource)
                  << (" pbo=") << (pbo) << (std::endl);
      assert((cudaSuccess) == (r));
    };
    while (!(glfwWindowShouldClose(window))) {
      glfwPollEvents();
      auto time = glfwGetTime();
      glClear(GL_COLOR_BUFFER_BIT);
      render(d_temp, width, height, bc);
      draw_texture(width, height);
      glfwSwapBuffers(window);
    }
    if (pbo) {
      cudaGraphicsUnregisterResource(g_cuda_pbo_resource);
      (std::cout) << (((std::chrono::high_resolution_clock::now()
                            .time_since_epoch()
                            .count()) -
                       (g_start)))
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__)
                  << (" cudaGraphicsUnregisterResource(g_cuda_pbo_resource) ")
                  << (std::endl);
      glDeleteBuffers(1, &pbo);
      (std::cout) << (((std::chrono::high_resolution_clock::now()
                            .time_since_epoch()
                            .count()) -
                       (g_start)))
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__) << (" glDeleteBuffers(1, &pbo) ")
                  << (std::endl);
      glDeleteTextures(1, &tex);
      (std::cout) << (((std::chrono::high_resolution_clock::now()
                            .time_since_epoch()
                            .count()) -
                       (g_start)))
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__) << (" glDeleteTextures(1, &tex) ")
                  << (std::endl);
    };
    cudaFree(d_temp);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (g_start)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" cudaFree(d_temp) ") << (std::endl);
    glfwDestroyWindow(window);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (g_start)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" glfwDestroyWindow(window) ") << (std::endl);
  };
  glfwTerminate();
  (std::cout) << (((std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count()) -
                   (g_start)))
              << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
              << (__func__) << (" glfwTerminate() ") << (std::endl);
}