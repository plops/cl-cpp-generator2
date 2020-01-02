
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;
void uploadTex(const void *image, int w, int h) {
  glGenTextures(1, &(state._fontTex));
  glBindTexture(GL_TEXTURE_2D, state._fontTex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
               image);
}
void initDraw() {
  glEnable(GL_TEXTURE_2D);
  glClearColor(0, 0, 0, 1);
}
void cleanupDraw() { glDeleteTextures(1, &(state._fontTex)); }
void drawFrame() {
  glClear(GL_COLOR_BUFFER_BIT);
  glBegin(GL_QUADS);
  glVertex2f((-5.e-1f), (-5.e-1f));
  glTexCoord2f(0, 0);
  glVertex2f((-5.e-1f), (5.e-1f));
  glTexCoord2f(0, 1);
  glVertex2f((5.e-1f), (5.e-1f));
  glTexCoord2f(1, 1);
  glVertex2f((5.e-1f), (-5.e-1f));
  glTexCoord2f(1, 0);
  glEnd();
  glfwSwapBuffers(state._window);
};