
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

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("screen") << (" ")
      << (std::setw(8)) << (" state._screen_offset[0]='")
      << (state._screen_offset[0]) << ("'") << (std::setw(8))
      << (" state._screen_offset[1]='") << (state._screen_offset[1]) << ("'")
      << (std::setw(8)) << (" state._screen_start_pan[0]='")
      << (state._screen_start_pan[0]) << ("'") << (std::setw(8))
      << (" state._screen_start_pan[1]='") << (state._screen_start_pan[1])
      << ("'") << (std::setw(8)) << (" state._screen_scale='")
      << (state._screen_scale) << ("'") << (std::setw(8))
      << (" state._screen_grid='") << (state._screen_grid) << ("'")
      << (std::endl) << (std::flush);
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
      glOrtho((0.f), width, height, (0.f), (-1.0f), (1.0f));
      // default offset to middle of screen
      state._screen_offset = {
          ((static_cast<float>(((screen_width()) / (-2)))) /
           (state._screen_scale)),
          ((static_cast<float>(((screen_height()) / (-2)))) /
           (state._screen_scale))};
      glMatrixMode(GL_MODELVIEW);
      glPushMatrix();
      glLoadIdentity();
    };
  };
  glClear(((GL_COLOR_BUFFER_BIT) | (GL_DEPTH_BUFFER_BIT)));
  auto mouse_state =
      glfwGetMouseButton(state._window, GLFW_MOUSE_BUTTON_MIDDLE);
  static int old_mouse_state = GLFW_RELEASE;
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
  auto mouse_before_zoom = glm::vec2();
  screen_to_world(static_cast<int>(mouse_pos[0]),
                  static_cast<int>(mouse_pos[1]), mouse_before_zoom);
  {
    auto key_state = glfwGetKey(state._window, GLFW_KEY_PERIOD);
    if ((key_state) == (GLFW_PRESS)) {
      state._screen_scale = (((0.90f)) * (state._screen_scale));
    };
  };
  {
    auto key_state = glfwGetKey(state._window, GLFW_KEY_COMMA);
    if ((key_state) == (GLFW_PRESS)) {
      state._screen_scale = (((1.10f)) * (state._screen_scale));
    };
  };
  old_mouse_state = mouse_state;
  auto world_top_left = glm::vec2();
  auto world_bottom_right = glm::vec2();
  screen_to_world(0, 0, world_top_left);
  screen_to_world(screen_width(), screen_height(), world_bottom_right);
  world_top_left[0] = floor(world_top_left[0]);
  world_top_left[1] = floor(world_top_left[1]);
  world_bottom_right[0] = ceil(world_bottom_right[0]);
  world_bottom_right[1] = ceil(world_bottom_right[1]);
  auto sx = 0;
  auto sy = 0;
  auto ex = 0;
  auto ey = 0;
  world_to_screen({0, world_top_left[1]}, sx, sy);
  world_to_screen({0, world_bottom_right[1]}, ex, ey);
  glColor4f((0.80f), (0.30f), (0.30f), 1);
  glEnable(GL_LINE_STIPPLE);
  glLineStipple(1, 0xF0F0);
  glBegin(GL_LINES);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  glEnd();
  world_to_screen({world_top_left[0], 0}, sx, sy);
  world_to_screen({world_bottom_right[0], 0}, ex, ey);
  glColor4f((0.80f), (0.30f), (0.30f), 1);
  glBegin(GL_LINES);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  glEnd();
  glColor4f((0.30f), (0.30f), (0.30f), 1);
  glLineStipple(1, 0xAAAA);
  glBegin(GL_LINES);
  world_to_screen({world_top_left[0], -10}, sx, sy);
  world_to_screen({world_bottom_right[0], -10}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -9}, sx, sy);
  world_to_screen({world_bottom_right[0], -9}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -8}, sx, sy);
  world_to_screen({world_bottom_right[0], -8}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -7}, sx, sy);
  world_to_screen({world_bottom_right[0], -7}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -6}, sx, sy);
  world_to_screen({world_bottom_right[0], -6}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -5}, sx, sy);
  world_to_screen({world_bottom_right[0], -5}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -4}, sx, sy);
  world_to_screen({world_bottom_right[0], -4}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -3}, sx, sy);
  world_to_screen({world_bottom_right[0], -3}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -2}, sx, sy);
  world_to_screen({world_bottom_right[0], -2}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -1}, sx, sy);
  world_to_screen({world_bottom_right[0], -1}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 0}, sx, sy);
  world_to_screen({world_bottom_right[0], 0}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 1}, sx, sy);
  world_to_screen({world_bottom_right[0], 1}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 2}, sx, sy);
  world_to_screen({world_bottom_right[0], 2}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 3}, sx, sy);
  world_to_screen({world_bottom_right[0], 3}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 4}, sx, sy);
  world_to_screen({world_bottom_right[0], 4}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 5}, sx, sy);
  world_to_screen({world_bottom_right[0], 5}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 6}, sx, sy);
  world_to_screen({world_bottom_right[0], 6}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 7}, sx, sy);
  world_to_screen({world_bottom_right[0], 7}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 8}, sx, sy);
  world_to_screen({world_bottom_right[0], 8}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 9}, sx, sy);
  world_to_screen({world_bottom_right[0], 9}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 10}, sx, sy);
  world_to_screen({world_bottom_right[0], 10}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-10, world_top_left[1]}, sx, sy);
  world_to_screen({-10, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-9, world_top_left[1]}, sx, sy);
  world_to_screen({-9, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-8, world_top_left[1]}, sx, sy);
  world_to_screen({-8, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-7, world_top_left[1]}, sx, sy);
  world_to_screen({-7, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-6, world_top_left[1]}, sx, sy);
  world_to_screen({-6, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-5, world_top_left[1]}, sx, sy);
  world_to_screen({-5, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-4, world_top_left[1]}, sx, sy);
  world_to_screen({-4, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-3, world_top_left[1]}, sx, sy);
  world_to_screen({-3, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-2, world_top_left[1]}, sx, sy);
  world_to_screen({-2, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-1, world_top_left[1]}, sx, sy);
  world_to_screen({-1, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({0, world_top_left[1]}, sx, sy);
  world_to_screen({0, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({1, world_top_left[1]}, sx, sy);
  world_to_screen({1, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({2, world_top_left[1]}, sx, sy);
  world_to_screen({2, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({3, world_top_left[1]}, sx, sy);
  world_to_screen({3, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({4, world_top_left[1]}, sx, sy);
  world_to_screen({4, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({5, world_top_left[1]}, sx, sy);
  world_to_screen({5, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({6, world_top_left[1]}, sx, sy);
  world_to_screen({6, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({7, world_top_left[1]}, sx, sy);
  world_to_screen({7, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({8, world_top_left[1]}, sx, sy);
  world_to_screen({8, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({9, world_top_left[1]}, sx, sy);
  world_to_screen({9, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({10, world_top_left[1]}, sx, sy);
  world_to_screen({10, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  glDisable(GL_LINE_STIPPLE);
  glEnd();
  int width = 0;
  int height = 0;
  glfwGetFramebufferSize(state._window, &width, &height);
  glColor4f(1, 1, 1, 1);
  glBegin(GL_LINES);
  auto x = state._cursor_xpos;
  auto y = state._cursor_ypos;
  glVertex2d(x, 0);
  glVertex2d(x, width);
  glVertex2d(0, y);
  glVertex2d(height, y);
  glEnd();
};