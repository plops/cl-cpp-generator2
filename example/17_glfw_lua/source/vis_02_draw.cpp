
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;
#include <algorithm>

// initialize static varibles
float Shape::world_scale = (1.0f);
glm::vec2 Shape::world_offset = {0, 0};
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
void draw_circle(float sx, float sy, float rad) {
  glBegin(GL_LINE_LOOP);
  {
    auto arg = (0.f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (8.771930e-3);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (1.7543860e-2);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (2.631579e-2);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (3.508772e-2);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (4.3859650e-2);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (5.263158e-2);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (6.140351e-2);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (7.017544e-2);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (7.894737e-2);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (8.771930e-2);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (9.649122e-2);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.10526316f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.1140351f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.1228070f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.1315790f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.1403509f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.14912280f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.15789473f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.16666667f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.17543860f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.18421052f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.19298245f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.20175439f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.21052632f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.21929824f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.228070f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.2368421f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.2456140f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.2543860f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.2631579f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.271930f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.2807018f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.28947368f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.29824560f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.30701753f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.31578946f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.32456142f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.33333334f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.3421053f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.3508772f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.3596491f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.3684210f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.3771930f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.3859649f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.3947369f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.4035088f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.4122807f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.4210526f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.42982456f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.43859650f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.4473684f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.456140f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.4649123f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.4736842f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.4824561f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.4912281f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.50f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.5087720f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.5175439f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.5263158f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.5350877f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.543860f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.5526316f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.5614035f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.5701754f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.5789474f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.5877193f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.5964912f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.6052632f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.6140351f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.6228070f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.6315789f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.6403509f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.6491228f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.6578947f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.6666667f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.6754386f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.684211f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.6929824f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.7017544f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.7105263f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.7192982f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.728070f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.7368421f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.7456141f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.7543859f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.7631579f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.771930f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.7807018f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.7894737f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.7982456f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.8070176f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.815789f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.8245614f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.8333333f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.8421053f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.8508772f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.8596491f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.8684211f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.8771930f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.8859649f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.8947368f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.9035088f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.912281f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.9210526f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.9298246f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.9385965f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.9473684f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.956140f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.9649123f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.9736842f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.9824561f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  {
    auto arg = (0.9912280f);
    glVertex2f(((sx) + (((rad) * (sinf(((2) * (M_PI) * (arg))))))),
               ((sy) + (((rad) * (cosf(((2) * (M_PI) * (arg))))))));
  };
  glEnd();
}
void initDraw() {
  state._temp_shape = nullptr;
  state._selected_node = nullptr;
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
  auto zoom_speed = (5.00e-2);
  screen_to_world(static_cast<int>(mouse_pos[0]),
                  static_cast<int>(mouse_pos[1]), mouse_before_zoom);
  {
    auto key_state = glfwGetKey(state._window, GLFW_KEY_PERIOD);
    if ((key_state) == (GLFW_PRESS)) {
      state._screen_scale =
          (((((1.0f)) - (zoom_speed))) * (state._screen_scale));
    };
  };
  {
    auto key_state = glfwGetKey(state._window, GLFW_KEY_COMMA);
    if ((key_state) == (GLFW_PRESS)) {
      state._screen_scale =
          (((((1.0f)) + (zoom_speed))) * (state._screen_scale));
    };
  };
  auto mouse_after_zoom = glm::vec2();
  screen_to_world(static_cast<int>(mouse_pos[0]),
                  static_cast<int>(mouse_pos[1]), mouse_after_zoom);
  (state._screen_offset) += (((mouse_before_zoom) - (mouse_after_zoom)));
  // compute snapped world cursor
  state._snapped_world_cursor[0] =
      floorf((((((0.50f)) + (mouse_after_zoom[0]))) * (state._screen_grid)));
  state._snapped_world_cursor[1] =
      floorf((((((0.50f)) + (mouse_after_zoom[1]))) * (state._screen_grid)));
  {
    // draw line
    {
      auto key_state = glfwGetKey(state._window, GLFW_KEY_L);
      if ((key_state) == (GLFW_PRESS)) {
        state._temp_shape = new Line();
        state._selected_node =
            state._temp_shape->get_next_node(state._snapped_world_cursor);
        state._selected_node =
            state._temp_shape->get_next_node(state._snapped_world_cursor);
      };
    };
    {
      auto key_state = glfwGetKey(state._window, GLFW_KEY_B);
      if ((key_state) == (GLFW_PRESS)) {
        state._temp_shape = new Box();
        state._selected_node =
            state._temp_shape->get_next_node(state._snapped_world_cursor);
        state._selected_node =
            state._temp_shape->get_next_node(state._snapped_world_cursor);
      };
    };
    {
      auto key_state = glfwGetKey(state._window, GLFW_KEY_C);
      if ((key_state) == (GLFW_PRESS)) {
        state._temp_shape = new Circle();
        state._selected_node =
            state._temp_shape->get_next_node(state._snapped_world_cursor);
        state._selected_node =
            state._temp_shape->get_next_node(state._snapped_world_cursor);
      };
    };
    {
      auto key_state = glfwGetKey(state._window, GLFW_KEY_M);
      if ((key_state) == (GLFW_PRESS)) {
        state._selected_node = nullptr;
        for (auto &shape : state._shapes) {
          state._selected_node = shape->hit_node(state._snapped_world_cursor);
          if (!((nullptr) == (state._selected_node))) {
            break;
          };
        };
      };
    };
    if (!((state._selected_node) == (nullptr))) {
      state._selected_node->pos = state._snapped_world_cursor;
    };
    auto left_mouse_button_state =
        glfwGetMouseButton(state._window, GLFW_MOUSE_BUTTON_LEFT);
    static int old_left_mouse_button_state = GLFW_RELEASE;
    if ((((old_left_mouse_button_state) == (GLFW_PRESS)) &&
         ((left_mouse_button_state) == (GLFW_RELEASE)))) {
      if (!((nullptr) == (state._temp_shape))) {
        state._selected_node =
            state._temp_shape->get_next_node(state._snapped_world_cursor);
        if ((nullptr) == (state._selected_node)) {
          //  shape is complete
          state._temp_shape->color = glm::vec4((1.0f), (1.0f), (1.0f), (1.0f));
          state._shapes.push_back(state._temp_shape);
        };
      };
    };
    old_left_mouse_button_state = left_mouse_button_state;
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
  world_to_screen({world_top_left[0], -100}, sx, sy);
  world_to_screen({world_bottom_right[0], -100}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -99}, sx, sy);
  world_to_screen({world_bottom_right[0], -99}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -98}, sx, sy);
  world_to_screen({world_bottom_right[0], -98}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -97}, sx, sy);
  world_to_screen({world_bottom_right[0], -97}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -96}, sx, sy);
  world_to_screen({world_bottom_right[0], -96}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -95}, sx, sy);
  world_to_screen({world_bottom_right[0], -95}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -94}, sx, sy);
  world_to_screen({world_bottom_right[0], -94}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -93}, sx, sy);
  world_to_screen({world_bottom_right[0], -93}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -92}, sx, sy);
  world_to_screen({world_bottom_right[0], -92}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -91}, sx, sy);
  world_to_screen({world_bottom_right[0], -91}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -90}, sx, sy);
  world_to_screen({world_bottom_right[0], -90}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -89}, sx, sy);
  world_to_screen({world_bottom_right[0], -89}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -88}, sx, sy);
  world_to_screen({world_bottom_right[0], -88}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -87}, sx, sy);
  world_to_screen({world_bottom_right[0], -87}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -86}, sx, sy);
  world_to_screen({world_bottom_right[0], -86}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -85}, sx, sy);
  world_to_screen({world_bottom_right[0], -85}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -84}, sx, sy);
  world_to_screen({world_bottom_right[0], -84}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -83}, sx, sy);
  world_to_screen({world_bottom_right[0], -83}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -82}, sx, sy);
  world_to_screen({world_bottom_right[0], -82}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -81}, sx, sy);
  world_to_screen({world_bottom_right[0], -81}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -80}, sx, sy);
  world_to_screen({world_bottom_right[0], -80}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -79}, sx, sy);
  world_to_screen({world_bottom_right[0], -79}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -78}, sx, sy);
  world_to_screen({world_bottom_right[0], -78}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -77}, sx, sy);
  world_to_screen({world_bottom_right[0], -77}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -76}, sx, sy);
  world_to_screen({world_bottom_right[0], -76}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -75}, sx, sy);
  world_to_screen({world_bottom_right[0], -75}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -74}, sx, sy);
  world_to_screen({world_bottom_right[0], -74}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -73}, sx, sy);
  world_to_screen({world_bottom_right[0], -73}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -72}, sx, sy);
  world_to_screen({world_bottom_right[0], -72}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -71}, sx, sy);
  world_to_screen({world_bottom_right[0], -71}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -70}, sx, sy);
  world_to_screen({world_bottom_right[0], -70}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -69}, sx, sy);
  world_to_screen({world_bottom_right[0], -69}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -68}, sx, sy);
  world_to_screen({world_bottom_right[0], -68}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -67}, sx, sy);
  world_to_screen({world_bottom_right[0], -67}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -66}, sx, sy);
  world_to_screen({world_bottom_right[0], -66}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -65}, sx, sy);
  world_to_screen({world_bottom_right[0], -65}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -64}, sx, sy);
  world_to_screen({world_bottom_right[0], -64}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -63}, sx, sy);
  world_to_screen({world_bottom_right[0], -63}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -62}, sx, sy);
  world_to_screen({world_bottom_right[0], -62}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -61}, sx, sy);
  world_to_screen({world_bottom_right[0], -61}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -60}, sx, sy);
  world_to_screen({world_bottom_right[0], -60}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -59}, sx, sy);
  world_to_screen({world_bottom_right[0], -59}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -58}, sx, sy);
  world_to_screen({world_bottom_right[0], -58}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -57}, sx, sy);
  world_to_screen({world_bottom_right[0], -57}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -56}, sx, sy);
  world_to_screen({world_bottom_right[0], -56}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -55}, sx, sy);
  world_to_screen({world_bottom_right[0], -55}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -54}, sx, sy);
  world_to_screen({world_bottom_right[0], -54}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -53}, sx, sy);
  world_to_screen({world_bottom_right[0], -53}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -52}, sx, sy);
  world_to_screen({world_bottom_right[0], -52}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -51}, sx, sy);
  world_to_screen({world_bottom_right[0], -51}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -50}, sx, sy);
  world_to_screen({world_bottom_right[0], -50}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -49}, sx, sy);
  world_to_screen({world_bottom_right[0], -49}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -48}, sx, sy);
  world_to_screen({world_bottom_right[0], -48}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -47}, sx, sy);
  world_to_screen({world_bottom_right[0], -47}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -46}, sx, sy);
  world_to_screen({world_bottom_right[0], -46}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -45}, sx, sy);
  world_to_screen({world_bottom_right[0], -45}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -44}, sx, sy);
  world_to_screen({world_bottom_right[0], -44}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -43}, sx, sy);
  world_to_screen({world_bottom_right[0], -43}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -42}, sx, sy);
  world_to_screen({world_bottom_right[0], -42}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -41}, sx, sy);
  world_to_screen({world_bottom_right[0], -41}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -40}, sx, sy);
  world_to_screen({world_bottom_right[0], -40}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -39}, sx, sy);
  world_to_screen({world_bottom_right[0], -39}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -38}, sx, sy);
  world_to_screen({world_bottom_right[0], -38}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -37}, sx, sy);
  world_to_screen({world_bottom_right[0], -37}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -36}, sx, sy);
  world_to_screen({world_bottom_right[0], -36}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -35}, sx, sy);
  world_to_screen({world_bottom_right[0], -35}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -34}, sx, sy);
  world_to_screen({world_bottom_right[0], -34}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -33}, sx, sy);
  world_to_screen({world_bottom_right[0], -33}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -32}, sx, sy);
  world_to_screen({world_bottom_right[0], -32}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -31}, sx, sy);
  world_to_screen({world_bottom_right[0], -31}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -30}, sx, sy);
  world_to_screen({world_bottom_right[0], -30}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -29}, sx, sy);
  world_to_screen({world_bottom_right[0], -29}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -28}, sx, sy);
  world_to_screen({world_bottom_right[0], -28}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -27}, sx, sy);
  world_to_screen({world_bottom_right[0], -27}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -26}, sx, sy);
  world_to_screen({world_bottom_right[0], -26}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -25}, sx, sy);
  world_to_screen({world_bottom_right[0], -25}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -24}, sx, sy);
  world_to_screen({world_bottom_right[0], -24}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -23}, sx, sy);
  world_to_screen({world_bottom_right[0], -23}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -22}, sx, sy);
  world_to_screen({world_bottom_right[0], -22}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -21}, sx, sy);
  world_to_screen({world_bottom_right[0], -21}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -20}, sx, sy);
  world_to_screen({world_bottom_right[0], -20}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -19}, sx, sy);
  world_to_screen({world_bottom_right[0], -19}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -18}, sx, sy);
  world_to_screen({world_bottom_right[0], -18}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -17}, sx, sy);
  world_to_screen({world_bottom_right[0], -17}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -16}, sx, sy);
  world_to_screen({world_bottom_right[0], -16}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -15}, sx, sy);
  world_to_screen({world_bottom_right[0], -15}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -14}, sx, sy);
  world_to_screen({world_bottom_right[0], -14}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -13}, sx, sy);
  world_to_screen({world_bottom_right[0], -13}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -12}, sx, sy);
  world_to_screen({world_bottom_right[0], -12}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], -11}, sx, sy);
  world_to_screen({world_bottom_right[0], -11}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
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
  world_to_screen({world_top_left[0], 11}, sx, sy);
  world_to_screen({world_bottom_right[0], 11}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 12}, sx, sy);
  world_to_screen({world_bottom_right[0], 12}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 13}, sx, sy);
  world_to_screen({world_bottom_right[0], 13}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 14}, sx, sy);
  world_to_screen({world_bottom_right[0], 14}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 15}, sx, sy);
  world_to_screen({world_bottom_right[0], 15}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 16}, sx, sy);
  world_to_screen({world_bottom_right[0], 16}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 17}, sx, sy);
  world_to_screen({world_bottom_right[0], 17}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 18}, sx, sy);
  world_to_screen({world_bottom_right[0], 18}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 19}, sx, sy);
  world_to_screen({world_bottom_right[0], 19}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 20}, sx, sy);
  world_to_screen({world_bottom_right[0], 20}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 21}, sx, sy);
  world_to_screen({world_bottom_right[0], 21}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 22}, sx, sy);
  world_to_screen({world_bottom_right[0], 22}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 23}, sx, sy);
  world_to_screen({world_bottom_right[0], 23}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 24}, sx, sy);
  world_to_screen({world_bottom_right[0], 24}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 25}, sx, sy);
  world_to_screen({world_bottom_right[0], 25}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 26}, sx, sy);
  world_to_screen({world_bottom_right[0], 26}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 27}, sx, sy);
  world_to_screen({world_bottom_right[0], 27}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 28}, sx, sy);
  world_to_screen({world_bottom_right[0], 28}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 29}, sx, sy);
  world_to_screen({world_bottom_right[0], 29}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 30}, sx, sy);
  world_to_screen({world_bottom_right[0], 30}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 31}, sx, sy);
  world_to_screen({world_bottom_right[0], 31}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 32}, sx, sy);
  world_to_screen({world_bottom_right[0], 32}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 33}, sx, sy);
  world_to_screen({world_bottom_right[0], 33}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 34}, sx, sy);
  world_to_screen({world_bottom_right[0], 34}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 35}, sx, sy);
  world_to_screen({world_bottom_right[0], 35}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 36}, sx, sy);
  world_to_screen({world_bottom_right[0], 36}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 37}, sx, sy);
  world_to_screen({world_bottom_right[0], 37}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 38}, sx, sy);
  world_to_screen({world_bottom_right[0], 38}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 39}, sx, sy);
  world_to_screen({world_bottom_right[0], 39}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 40}, sx, sy);
  world_to_screen({world_bottom_right[0], 40}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 41}, sx, sy);
  world_to_screen({world_bottom_right[0], 41}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 42}, sx, sy);
  world_to_screen({world_bottom_right[0], 42}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 43}, sx, sy);
  world_to_screen({world_bottom_right[0], 43}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 44}, sx, sy);
  world_to_screen({world_bottom_right[0], 44}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 45}, sx, sy);
  world_to_screen({world_bottom_right[0], 45}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 46}, sx, sy);
  world_to_screen({world_bottom_right[0], 46}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 47}, sx, sy);
  world_to_screen({world_bottom_right[0], 47}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 48}, sx, sy);
  world_to_screen({world_bottom_right[0], 48}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 49}, sx, sy);
  world_to_screen({world_bottom_right[0], 49}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 50}, sx, sy);
  world_to_screen({world_bottom_right[0], 50}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 51}, sx, sy);
  world_to_screen({world_bottom_right[0], 51}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 52}, sx, sy);
  world_to_screen({world_bottom_right[0], 52}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 53}, sx, sy);
  world_to_screen({world_bottom_right[0], 53}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 54}, sx, sy);
  world_to_screen({world_bottom_right[0], 54}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 55}, sx, sy);
  world_to_screen({world_bottom_right[0], 55}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 56}, sx, sy);
  world_to_screen({world_bottom_right[0], 56}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 57}, sx, sy);
  world_to_screen({world_bottom_right[0], 57}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 58}, sx, sy);
  world_to_screen({world_bottom_right[0], 58}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 59}, sx, sy);
  world_to_screen({world_bottom_right[0], 59}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 60}, sx, sy);
  world_to_screen({world_bottom_right[0], 60}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 61}, sx, sy);
  world_to_screen({world_bottom_right[0], 61}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 62}, sx, sy);
  world_to_screen({world_bottom_right[0], 62}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 63}, sx, sy);
  world_to_screen({world_bottom_right[0], 63}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 64}, sx, sy);
  world_to_screen({world_bottom_right[0], 64}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 65}, sx, sy);
  world_to_screen({world_bottom_right[0], 65}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 66}, sx, sy);
  world_to_screen({world_bottom_right[0], 66}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 67}, sx, sy);
  world_to_screen({world_bottom_right[0], 67}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 68}, sx, sy);
  world_to_screen({world_bottom_right[0], 68}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 69}, sx, sy);
  world_to_screen({world_bottom_right[0], 69}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 70}, sx, sy);
  world_to_screen({world_bottom_right[0], 70}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 71}, sx, sy);
  world_to_screen({world_bottom_right[0], 71}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 72}, sx, sy);
  world_to_screen({world_bottom_right[0], 72}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 73}, sx, sy);
  world_to_screen({world_bottom_right[0], 73}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 74}, sx, sy);
  world_to_screen({world_bottom_right[0], 74}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 75}, sx, sy);
  world_to_screen({world_bottom_right[0], 75}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 76}, sx, sy);
  world_to_screen({world_bottom_right[0], 76}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 77}, sx, sy);
  world_to_screen({world_bottom_right[0], 77}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 78}, sx, sy);
  world_to_screen({world_bottom_right[0], 78}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 79}, sx, sy);
  world_to_screen({world_bottom_right[0], 79}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 80}, sx, sy);
  world_to_screen({world_bottom_right[0], 80}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 81}, sx, sy);
  world_to_screen({world_bottom_right[0], 81}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 82}, sx, sy);
  world_to_screen({world_bottom_right[0], 82}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 83}, sx, sy);
  world_to_screen({world_bottom_right[0], 83}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 84}, sx, sy);
  world_to_screen({world_bottom_right[0], 84}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 85}, sx, sy);
  world_to_screen({world_bottom_right[0], 85}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 86}, sx, sy);
  world_to_screen({world_bottom_right[0], 86}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 87}, sx, sy);
  world_to_screen({world_bottom_right[0], 87}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 88}, sx, sy);
  world_to_screen({world_bottom_right[0], 88}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 89}, sx, sy);
  world_to_screen({world_bottom_right[0], 89}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 90}, sx, sy);
  world_to_screen({world_bottom_right[0], 90}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 91}, sx, sy);
  world_to_screen({world_bottom_right[0], 91}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 92}, sx, sy);
  world_to_screen({world_bottom_right[0], 92}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 93}, sx, sy);
  world_to_screen({world_bottom_right[0], 93}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 94}, sx, sy);
  world_to_screen({world_bottom_right[0], 94}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 95}, sx, sy);
  world_to_screen({world_bottom_right[0], 95}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 96}, sx, sy);
  world_to_screen({world_bottom_right[0], 96}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 97}, sx, sy);
  world_to_screen({world_bottom_right[0], 97}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 98}, sx, sy);
  world_to_screen({world_bottom_right[0], 98}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 99}, sx, sy);
  world_to_screen({world_bottom_right[0], 99}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({world_top_left[0], 100}, sx, sy);
  world_to_screen({world_bottom_right[0], 100}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-100, world_top_left[1]}, sx, sy);
  world_to_screen({-100, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-99, world_top_left[1]}, sx, sy);
  world_to_screen({-99, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-98, world_top_left[1]}, sx, sy);
  world_to_screen({-98, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-97, world_top_left[1]}, sx, sy);
  world_to_screen({-97, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-96, world_top_left[1]}, sx, sy);
  world_to_screen({-96, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-95, world_top_left[1]}, sx, sy);
  world_to_screen({-95, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-94, world_top_left[1]}, sx, sy);
  world_to_screen({-94, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-93, world_top_left[1]}, sx, sy);
  world_to_screen({-93, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-92, world_top_left[1]}, sx, sy);
  world_to_screen({-92, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-91, world_top_left[1]}, sx, sy);
  world_to_screen({-91, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-90, world_top_left[1]}, sx, sy);
  world_to_screen({-90, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-89, world_top_left[1]}, sx, sy);
  world_to_screen({-89, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-88, world_top_left[1]}, sx, sy);
  world_to_screen({-88, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-87, world_top_left[1]}, sx, sy);
  world_to_screen({-87, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-86, world_top_left[1]}, sx, sy);
  world_to_screen({-86, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-85, world_top_left[1]}, sx, sy);
  world_to_screen({-85, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-84, world_top_left[1]}, sx, sy);
  world_to_screen({-84, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-83, world_top_left[1]}, sx, sy);
  world_to_screen({-83, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-82, world_top_left[1]}, sx, sy);
  world_to_screen({-82, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-81, world_top_left[1]}, sx, sy);
  world_to_screen({-81, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-80, world_top_left[1]}, sx, sy);
  world_to_screen({-80, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-79, world_top_left[1]}, sx, sy);
  world_to_screen({-79, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-78, world_top_left[1]}, sx, sy);
  world_to_screen({-78, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-77, world_top_left[1]}, sx, sy);
  world_to_screen({-77, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-76, world_top_left[1]}, sx, sy);
  world_to_screen({-76, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-75, world_top_left[1]}, sx, sy);
  world_to_screen({-75, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-74, world_top_left[1]}, sx, sy);
  world_to_screen({-74, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-73, world_top_left[1]}, sx, sy);
  world_to_screen({-73, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-72, world_top_left[1]}, sx, sy);
  world_to_screen({-72, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-71, world_top_left[1]}, sx, sy);
  world_to_screen({-71, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-70, world_top_left[1]}, sx, sy);
  world_to_screen({-70, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-69, world_top_left[1]}, sx, sy);
  world_to_screen({-69, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-68, world_top_left[1]}, sx, sy);
  world_to_screen({-68, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-67, world_top_left[1]}, sx, sy);
  world_to_screen({-67, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-66, world_top_left[1]}, sx, sy);
  world_to_screen({-66, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-65, world_top_left[1]}, sx, sy);
  world_to_screen({-65, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-64, world_top_left[1]}, sx, sy);
  world_to_screen({-64, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-63, world_top_left[1]}, sx, sy);
  world_to_screen({-63, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-62, world_top_left[1]}, sx, sy);
  world_to_screen({-62, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-61, world_top_left[1]}, sx, sy);
  world_to_screen({-61, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-60, world_top_left[1]}, sx, sy);
  world_to_screen({-60, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-59, world_top_left[1]}, sx, sy);
  world_to_screen({-59, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-58, world_top_left[1]}, sx, sy);
  world_to_screen({-58, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-57, world_top_left[1]}, sx, sy);
  world_to_screen({-57, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-56, world_top_left[1]}, sx, sy);
  world_to_screen({-56, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-55, world_top_left[1]}, sx, sy);
  world_to_screen({-55, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-54, world_top_left[1]}, sx, sy);
  world_to_screen({-54, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-53, world_top_left[1]}, sx, sy);
  world_to_screen({-53, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-52, world_top_left[1]}, sx, sy);
  world_to_screen({-52, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-51, world_top_left[1]}, sx, sy);
  world_to_screen({-51, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-50, world_top_left[1]}, sx, sy);
  world_to_screen({-50, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-49, world_top_left[1]}, sx, sy);
  world_to_screen({-49, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-48, world_top_left[1]}, sx, sy);
  world_to_screen({-48, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-47, world_top_left[1]}, sx, sy);
  world_to_screen({-47, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-46, world_top_left[1]}, sx, sy);
  world_to_screen({-46, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-45, world_top_left[1]}, sx, sy);
  world_to_screen({-45, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-44, world_top_left[1]}, sx, sy);
  world_to_screen({-44, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-43, world_top_left[1]}, sx, sy);
  world_to_screen({-43, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-42, world_top_left[1]}, sx, sy);
  world_to_screen({-42, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-41, world_top_left[1]}, sx, sy);
  world_to_screen({-41, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-40, world_top_left[1]}, sx, sy);
  world_to_screen({-40, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-39, world_top_left[1]}, sx, sy);
  world_to_screen({-39, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-38, world_top_left[1]}, sx, sy);
  world_to_screen({-38, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-37, world_top_left[1]}, sx, sy);
  world_to_screen({-37, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-36, world_top_left[1]}, sx, sy);
  world_to_screen({-36, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-35, world_top_left[1]}, sx, sy);
  world_to_screen({-35, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-34, world_top_left[1]}, sx, sy);
  world_to_screen({-34, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-33, world_top_left[1]}, sx, sy);
  world_to_screen({-33, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-32, world_top_left[1]}, sx, sy);
  world_to_screen({-32, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-31, world_top_left[1]}, sx, sy);
  world_to_screen({-31, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-30, world_top_left[1]}, sx, sy);
  world_to_screen({-30, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-29, world_top_left[1]}, sx, sy);
  world_to_screen({-29, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-28, world_top_left[1]}, sx, sy);
  world_to_screen({-28, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-27, world_top_left[1]}, sx, sy);
  world_to_screen({-27, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-26, world_top_left[1]}, sx, sy);
  world_to_screen({-26, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-25, world_top_left[1]}, sx, sy);
  world_to_screen({-25, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-24, world_top_left[1]}, sx, sy);
  world_to_screen({-24, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-23, world_top_left[1]}, sx, sy);
  world_to_screen({-23, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-22, world_top_left[1]}, sx, sy);
  world_to_screen({-22, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-21, world_top_left[1]}, sx, sy);
  world_to_screen({-21, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-20, world_top_left[1]}, sx, sy);
  world_to_screen({-20, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-19, world_top_left[1]}, sx, sy);
  world_to_screen({-19, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-18, world_top_left[1]}, sx, sy);
  world_to_screen({-18, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-17, world_top_left[1]}, sx, sy);
  world_to_screen({-17, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-16, world_top_left[1]}, sx, sy);
  world_to_screen({-16, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-15, world_top_left[1]}, sx, sy);
  world_to_screen({-15, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-14, world_top_left[1]}, sx, sy);
  world_to_screen({-14, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-13, world_top_left[1]}, sx, sy);
  world_to_screen({-13, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-12, world_top_left[1]}, sx, sy);
  world_to_screen({-12, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({-11, world_top_left[1]}, sx, sy);
  world_to_screen({-11, world_bottom_right[1]}, ex, ey);
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
  world_to_screen({11, world_top_left[1]}, sx, sy);
  world_to_screen({11, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({12, world_top_left[1]}, sx, sy);
  world_to_screen({12, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({13, world_top_left[1]}, sx, sy);
  world_to_screen({13, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({14, world_top_left[1]}, sx, sy);
  world_to_screen({14, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({15, world_top_left[1]}, sx, sy);
  world_to_screen({15, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({16, world_top_left[1]}, sx, sy);
  world_to_screen({16, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({17, world_top_left[1]}, sx, sy);
  world_to_screen({17, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({18, world_top_left[1]}, sx, sy);
  world_to_screen({18, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({19, world_top_left[1]}, sx, sy);
  world_to_screen({19, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({20, world_top_left[1]}, sx, sy);
  world_to_screen({20, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({21, world_top_left[1]}, sx, sy);
  world_to_screen({21, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({22, world_top_left[1]}, sx, sy);
  world_to_screen({22, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({23, world_top_left[1]}, sx, sy);
  world_to_screen({23, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({24, world_top_left[1]}, sx, sy);
  world_to_screen({24, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({25, world_top_left[1]}, sx, sy);
  world_to_screen({25, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({26, world_top_left[1]}, sx, sy);
  world_to_screen({26, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({27, world_top_left[1]}, sx, sy);
  world_to_screen({27, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({28, world_top_left[1]}, sx, sy);
  world_to_screen({28, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({29, world_top_left[1]}, sx, sy);
  world_to_screen({29, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({30, world_top_left[1]}, sx, sy);
  world_to_screen({30, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({31, world_top_left[1]}, sx, sy);
  world_to_screen({31, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({32, world_top_left[1]}, sx, sy);
  world_to_screen({32, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({33, world_top_left[1]}, sx, sy);
  world_to_screen({33, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({34, world_top_left[1]}, sx, sy);
  world_to_screen({34, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({35, world_top_left[1]}, sx, sy);
  world_to_screen({35, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({36, world_top_left[1]}, sx, sy);
  world_to_screen({36, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({37, world_top_left[1]}, sx, sy);
  world_to_screen({37, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({38, world_top_left[1]}, sx, sy);
  world_to_screen({38, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({39, world_top_left[1]}, sx, sy);
  world_to_screen({39, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({40, world_top_left[1]}, sx, sy);
  world_to_screen({40, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({41, world_top_left[1]}, sx, sy);
  world_to_screen({41, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({42, world_top_left[1]}, sx, sy);
  world_to_screen({42, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({43, world_top_left[1]}, sx, sy);
  world_to_screen({43, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({44, world_top_left[1]}, sx, sy);
  world_to_screen({44, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({45, world_top_left[1]}, sx, sy);
  world_to_screen({45, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({46, world_top_left[1]}, sx, sy);
  world_to_screen({46, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({47, world_top_left[1]}, sx, sy);
  world_to_screen({47, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({48, world_top_left[1]}, sx, sy);
  world_to_screen({48, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({49, world_top_left[1]}, sx, sy);
  world_to_screen({49, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({50, world_top_left[1]}, sx, sy);
  world_to_screen({50, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({51, world_top_left[1]}, sx, sy);
  world_to_screen({51, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({52, world_top_left[1]}, sx, sy);
  world_to_screen({52, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({53, world_top_left[1]}, sx, sy);
  world_to_screen({53, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({54, world_top_left[1]}, sx, sy);
  world_to_screen({54, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({55, world_top_left[1]}, sx, sy);
  world_to_screen({55, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({56, world_top_left[1]}, sx, sy);
  world_to_screen({56, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({57, world_top_left[1]}, sx, sy);
  world_to_screen({57, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({58, world_top_left[1]}, sx, sy);
  world_to_screen({58, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({59, world_top_left[1]}, sx, sy);
  world_to_screen({59, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({60, world_top_left[1]}, sx, sy);
  world_to_screen({60, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({61, world_top_left[1]}, sx, sy);
  world_to_screen({61, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({62, world_top_left[1]}, sx, sy);
  world_to_screen({62, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({63, world_top_left[1]}, sx, sy);
  world_to_screen({63, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({64, world_top_left[1]}, sx, sy);
  world_to_screen({64, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({65, world_top_left[1]}, sx, sy);
  world_to_screen({65, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({66, world_top_left[1]}, sx, sy);
  world_to_screen({66, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({67, world_top_left[1]}, sx, sy);
  world_to_screen({67, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({68, world_top_left[1]}, sx, sy);
  world_to_screen({68, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({69, world_top_left[1]}, sx, sy);
  world_to_screen({69, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({70, world_top_left[1]}, sx, sy);
  world_to_screen({70, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({71, world_top_left[1]}, sx, sy);
  world_to_screen({71, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({72, world_top_left[1]}, sx, sy);
  world_to_screen({72, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({73, world_top_left[1]}, sx, sy);
  world_to_screen({73, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({74, world_top_left[1]}, sx, sy);
  world_to_screen({74, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({75, world_top_left[1]}, sx, sy);
  world_to_screen({75, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({76, world_top_left[1]}, sx, sy);
  world_to_screen({76, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({77, world_top_left[1]}, sx, sy);
  world_to_screen({77, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({78, world_top_left[1]}, sx, sy);
  world_to_screen({78, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({79, world_top_left[1]}, sx, sy);
  world_to_screen({79, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({80, world_top_left[1]}, sx, sy);
  world_to_screen({80, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({81, world_top_left[1]}, sx, sy);
  world_to_screen({81, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({82, world_top_left[1]}, sx, sy);
  world_to_screen({82, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({83, world_top_left[1]}, sx, sy);
  world_to_screen({83, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({84, world_top_left[1]}, sx, sy);
  world_to_screen({84, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({85, world_top_left[1]}, sx, sy);
  world_to_screen({85, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({86, world_top_left[1]}, sx, sy);
  world_to_screen({86, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({87, world_top_left[1]}, sx, sy);
  world_to_screen({87, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({88, world_top_left[1]}, sx, sy);
  world_to_screen({88, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({89, world_top_left[1]}, sx, sy);
  world_to_screen({89, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({90, world_top_left[1]}, sx, sy);
  world_to_screen({90, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({91, world_top_left[1]}, sx, sy);
  world_to_screen({91, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({92, world_top_left[1]}, sx, sy);
  world_to_screen({92, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({93, world_top_left[1]}, sx, sy);
  world_to_screen({93, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({94, world_top_left[1]}, sx, sy);
  world_to_screen({94, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({95, world_top_left[1]}, sx, sy);
  world_to_screen({95, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({96, world_top_left[1]}, sx, sy);
  world_to_screen({96, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({97, world_top_left[1]}, sx, sy);
  world_to_screen({97, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({98, world_top_left[1]}, sx, sy);
  world_to_screen({98, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({99, world_top_left[1]}, sx, sy);
  world_to_screen({99, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  world_to_screen({100, world_top_left[1]}, sx, sy);
  world_to_screen({100, world_bottom_right[1]}, ex, ey);
  glVertex2f(sx, sy);
  glVertex2f(ex, ey);
  glEnd();
  glLineStipple(1, 0xFFFF);
  glDisable(GL_LINE_STIPPLE);
  // draw the geometric objects
  Shape::world_scale = state._screen_scale;
  Shape::world_offset = state._screen_offset;
  for (auto shape : state._shapes) {
    shape->draw();
    shape->draw_nodes();
  };
  if (!((nullptr) == (state._temp_shape))) {
    state._temp_shape->draw();
    state._temp_shape->draw_nodes();
  };
  // draw snapped cursor circle
  world_to_screen(state._snapped_world_cursor, sx, sy);
  glColor3f((1.0f), (1.0f), (0.f));
  draw_circle(sx, sy, 3);
  glColor4f(1, 1, 1, 1);
  glBegin(GL_LINES);
  auto x = state._cursor_xpos;
  auto y = state._cursor_ypos;
  auto h = screen_height();
  auto w = screen_width();
  glVertex2d(x, ((0) * (h)));
  glVertex2d(x, ((1) * (h)));
  glVertex2d(((0) * (w)), y);
  glVertex2d(((1) * (w)), y);
  glEnd();
};