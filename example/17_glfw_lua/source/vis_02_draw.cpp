
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
int screen_width() {
  int width = 0;
  int height = 0;
  glfwGetFramebufferSize(state._window, &width, &height);
  return width;
}
int screen_height() {
  int width = 0;
  int height = 0;
  glfwGetFramebufferSize(state._window, &width, &height);
  return height;
}
glm::vec2 get_mouse_position() {
  auto x = (0.);
  auto y = (0.);
  glfwGetCursorPos(state._window, &x, &y);
  return glm::vec2({static_cast<float>(x), static_cast<float>(y)});
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
  state._screen_offset = {(0.f), (0.f)};
  state._screen_start_pan = {(0.f), (0.f)};
  state._screen_scale = (10.f);
  state._screen_grid = (1.0f);
  // default offset to middle of screen
  state._screen_offset = {
      ((static_cast<float>(((screen_width()) / (-2)))) / (state._screen_scale)),
      ((static_cast<float>(((screen_height()) / (-2)))) /
       (state._screen_scale))};
}
void world_to_screen(const glm::vec2 &v, int &screeni, int &screenj) {
  screeni = static_cast<int>(
      ((((v[0]) - (state._screen_offset[0]))) * (state._screen_scale)));
  screenj = static_cast<int>(
      ((((v[1]) - (state._screen_offset[1]))) * (state._screen_scale)));
}
void screen_to_world(int screeni, int screenj, glm::vec2 &v) {
  v[0] = ((((static_cast<float>(screeni)) / (state._screen_scale))) +
          (state._screen_offset[0]));
  v[1] = ((((static_cast<float>(screenj)) / (state._screen_scale))) +
          (state._screen_offset[1]));
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
  auto mouse_state = glfwGetMouseButton(state._window, GLFW_MOUSE_BUTTON_LEFT);
  auto old_mouse_state = GLFW_RELEASE;
  auto mouse_pos = get_mouse_position();
  if ((((mouse_state) == (GLFW_PRESS)) &&
       ((old_mouse_state) == (GLFW_RELEASE)))) {
    state._screen_start_pan = mouse_pos;
  };
  if ((((mouse_state) == (GLFW_PRESS)) &&
       ((old_mouse_state) == (GLFW_PRESS)))) {
    (state._screen_offset) -=
        (((((mouse_pos) - (state._screen_start_pan))) / (state._screen_scale)));
    state._screen_start_pan = mouse_pos;
  };
  old_mouse_state = mouse_state;
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