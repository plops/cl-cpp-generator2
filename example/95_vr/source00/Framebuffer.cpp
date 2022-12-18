// no preamble
#include "Framebuffer.h"
#include "VrApi.h"
#include "VrApi_Helpers.h"
#include "VrApi_Input.h"
#include "VrApi_SystemUtils.h"
#include "android_native_app_glue.h"
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <android/log.h>
#include <android/window.h>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <vector>
Framebuffer::Framebuffer(GLsizei w, GLsizei h)
    : swap_chain_index(0), width(w), height(h),
      color_texture_swap_chain(vrapi_CreateTextureSwapChain3(
          VRAPI_TEXTURE_TYPE_2D, GL_RGBA8, w, h, 1, 3)) {
  if ((nullptr) == (color_texture_swap_chain)) {
    __android_log_print(ANDROID_LOG_VERBOSE, "hello_quest",
                        "cant create color texture swap chain");
    std::exit(-1);
  }
  swap_chain_length = vrapi_GetTextureSwapChainLength(color_texture_swap_chain);
  depth_renderbuffers.resize(swap_chain_length);
  glGenRenderbuffers(swap_chain_length, depth_renderbuffers.data());
  framebuffers.resize(swap_chain_length);
  glGenFramebuffers(swap_chain_length, framebuffers.data());
  for (auto i = 0; (i) < (swap_chain_length); (i) += (1)) {
    __android_log_print(ANDROID_LOG_VERBOSE, "hello_quest", "%s",
                        fmt::format("color texture   i='{}'", i).c_str());
    auto color_texture =
        vrapi_GetTextureSwapChainHandle(color_texture_swap_chain, i);
    glBindTexture(GL_TEXTURE_2D, color_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
    __android_log_print(ANDROID_LOG_VERBOSE, "hello_quest", "%s",
                        fmt::format("create depth buffer  i='{}'", i).c_str());
    glBindRenderbuffer(GL_RENDERBUFFER, depth_renderbuffers.at(i));
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    __android_log_print(ANDROID_LOG_VERBOSE, "hello_quest", "%s",
                        fmt::format("create framebuffer  i='{}'", i).c_str());
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffers.at(i));
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, color_texture, 0);
    glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                              GL_RENDERBUFFER, depth_renderbuffers.at(i));
    {
      auto status = glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER);
      if (!((GL_FRAMEBUFFER_COMPLETE) == (status))) {
        __android_log_print(
            ANDROID_LOG_VERBOSE, "hello_quest", "%s",
            fmt::format("cant create framebuffer  i='{}'", i).c_str());
        std::exit(-1);
      }
    }
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
  }
}
Framebuffer::~Framebuffer() {
  glDeleteFramebuffers(swap_chain_length, framebuffers.data());
  glDeleteRenderbuffers(swap_chain_length, depth_renderbuffers.data());
  vrapi_DestroyTextureSwapChain(color_texture_swap_chain);
}