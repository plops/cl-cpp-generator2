
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;
#include <algorithm>
void uploadTex(const void *image, int w, int h) {
  glGenTextures(1, &(state._fontTex));
  glBindTexture(GL_TEXTURE_2D, state._fontTex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
               image);
}
void initDraw() {
  {
    // no debug
    std::lock_guard<std::mutex> guard(state._draw_mutex);
    state._draw_display_log = true;
    state._draw_offset_x = (-3.00e-2);
    state._draw_offset_y = (-0.440f);
    state._draw_scale_x = (0.220f);
    state._draw_scale_y = (0.230f);
    state._draw_alpha = (0.190f);
    state._draw_marker_x = (1.00e+2);
  };
  glHint(GL_LINE_SMOOTH, GL_NICEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glClearColor(0, 0, 0, 1);
  state._framebufferResized = true;
}
void cleanupDraw() { glDeleteTextures(1, &(state._fontTex)); }
void drawFrame() {
  if (state._framebufferResized) {
    state._framebufferResized = false;
    int width = 0;
    int height = 0;
    while ((((0) == (width)) || ((0) == (height)))) {
      glfwGetFramebufferSize(state._window, &width, &height);
      glViewport(0, 0, width, height);
      glMatrixMode(GL_PROJECTION);
      glPushMatrix();
      glLoadIdentity();
      glOrtho((-1.0f), (1.0f), (-1.0f), (1.0f), (-1.0f), (1.0f));
      glMatrixMode(GL_MODELVIEW);
      glPushMatrix();
      glLoadIdentity();
    };
  };
  glClear(((GL_COLOR_BUFFER_BIT) | (GL_DEPTH_BUFFER_BIT)));
};