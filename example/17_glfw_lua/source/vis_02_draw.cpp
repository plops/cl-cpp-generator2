
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
  glColor4f((0.30f), (0.30f), (0.30f), 1);
  glBegin(GL_LINES);
  glVertex2f(-1, (-1.0f));
  glVertex2f(1, (-1.0f));
  glVertex2f(-1, (-0.90f));
  glVertex2f(1, (-0.90f));
  glVertex2f(-1, (-0.80f));
  glVertex2f(1, (-0.80f));
  glVertex2f(-1, (-0.70f));
  glVertex2f(1, (-0.70f));
  glVertex2f(-1, (-0.60f));
  glVertex2f(1, (-0.60f));
  glVertex2f(-1, (-0.50f));
  glVertex2f(1, (-0.50f));
  glVertex2f(-1, (-0.40f));
  glVertex2f(1, (-0.40f));
  glVertex2f(-1, (-0.30f));
  glVertex2f(1, (-0.30f));
  glVertex2f(-1, (-0.20f));
  glVertex2f(1, (-0.20f));
  glVertex2f(-1, (-0.10f));
  glVertex2f(1, (-0.10f));
  glVertex2f(-1, (0.f));
  glVertex2f(1, (0.f));
  glVertex2f(-1, (0.10f));
  glVertex2f(1, (0.10f));
  glVertex2f(-1, (0.20f));
  glVertex2f(1, (0.20f));
  glVertex2f(-1, (0.30f));
  glVertex2f(1, (0.30f));
  glVertex2f(-1, (0.40f));
  glVertex2f(1, (0.40f));
  glVertex2f(-1, (0.50f));
  glVertex2f(1, (0.50f));
  glVertex2f(-1, (0.60f));
  glVertex2f(1, (0.60f));
  glVertex2f(-1, (0.70f));
  glVertex2f(1, (0.70f));
  glVertex2f(-1, (0.80f));
  glVertex2f(1, (0.80f));
  glVertex2f(-1, (0.90f));
  glVertex2f(1, (0.90f));
  glVertex2f(-1, (1.0f));
  glVertex2f(1, (1.0f));
  glVertex2f((-1.0f), -1);
  glVertex2f((-1.0f), 1);
  glVertex2f((-0.90f), -1);
  glVertex2f((-0.90f), 1);
  glVertex2f((-0.80f), -1);
  glVertex2f((-0.80f), 1);
  glVertex2f((-0.70f), -1);
  glVertex2f((-0.70f), 1);
  glVertex2f((-0.60f), -1);
  glVertex2f((-0.60f), 1);
  glVertex2f((-0.50f), -1);
  glVertex2f((-0.50f), 1);
  glVertex2f((-0.40f), -1);
  glVertex2f((-0.40f), 1);
  glVertex2f((-0.30f), -1);
  glVertex2f((-0.30f), 1);
  glVertex2f((-0.20f), -1);
  glVertex2f((-0.20f), 1);
  glVertex2f((-0.10f), -1);
  glVertex2f((-0.10f), 1);
  glVertex2f((0.f), -1);
  glVertex2f((0.f), 1);
  glVertex2f((0.10f), -1);
  glVertex2f((0.10f), 1);
  glVertex2f((0.20f), -1);
  glVertex2f((0.20f), 1);
  glVertex2f((0.30f), -1);
  glVertex2f((0.30f), 1);
  glVertex2f((0.40f), -1);
  glVertex2f((0.40f), 1);
  glVertex2f((0.50f), -1);
  glVertex2f((0.50f), 1);
  glVertex2f((0.60f), -1);
  glVertex2f((0.60f), 1);
  glVertex2f((0.70f), -1);
  glVertex2f((0.70f), 1);
  glVertex2f((0.80f), -1);
  glVertex2f((0.80f), 1);
  glVertex2f((0.90f), -1);
  glVertex2f((0.90f), 1);
  glVertex2f((1.0f), -1);
  glVertex2f((1.0f), 1);
  glEnd();
  int width = 0;
  int height = 0;
  glfwGetFramebufferSize(state._window, &width, &height);
  glColor4f(1, 1, 1, 1);
  glBegin(GL_LINES);
  auto x = ((2) * (((((state._cursor_xpos) / (width))) - ((0.50f)))));
  auto y = ((-2) * (((((state._cursor_ypos) / (height))) - ((0.50f)))));
  glVertex2d(x, -1);
  glVertex2d(x, 1);
  glVertex2d(-1, y);
  glVertex2d(1, y);
  glEnd();
};