
#include "utils.h"

#include "globals.h"

extern State state;
// implementation

GLTexture::GLTexture(uint w, uint h, uint type) : width{w}, height{h} {
  glGenTextures(1, &ID);
  glBindTextures(GL_TEXTURE_2D, ID);
  switch (type) {
  case DEFAULT: {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_BGR,
                 GL_UNSIGNED_BYTE, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_MAG_FILTER, GL_NEAREST);
    break;
  };
  case INITTARGET: {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, 0);
    break;
  };
  case FLOAT: {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, FLOAT,
                 0);
    break;
  };
  }
  glBindTexture(GL_TEXTURE_2D, 0);
  CheckGL();
}
GLTexture::~GLTexture() {
  glDeleteTextures(1, &ID);
  CheckGL();
}
void GLTexture::Bind(const uint slot) {
  glActiveTexture((GL_TEXTURE0) + (slot));
  glBindTexture(GL_TEXTURE_2D, ID);
  CheckGL();
}
void GLTexture::CopyFrom(Surface *src) {
  glBindTexture(GL_TEXTURE_2D, ID);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, src->buffer);
  CheckGL();
}
void GLTexture::CopyTo(Surface *dst) {
  glBindTexture(GL_TEXTURE_2D, ID);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, dst->buffer);
  CheckGL();
}