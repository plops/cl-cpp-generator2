#ifndef VIS_04_GL_TEXTURE_H
#define VIS_04_GL_TEXTURE_H
#include "utils.h"
;
#include "globals.h"
;
// header
;
class GLTexture {
public:
  enum { DEFAULT = 0, FLOAT = 1, INITTARGET = 2 };
  GLTexture(uint w, uint h, uint type = DEFAULT);
  ~GLTexture();
  void Bind(const uint slot = 0);
  void CopyFrom(Surface *src);
  void CopyTo(Surface *dst);
  q;
  GLuint ID = 0;
  uint width = 0;
  uint height = 0;
};
#endif