#ifndef VIS_05_GL_SHADER_H
#define VIS_05_GL_SHADER_H
#include "utils.h"
;
#include "globals.h"
;
// header
;
class Shader {
public:
  Shader(const char *vfile, const char *pfile, bool fromString);
  ~Shader();
  void Init(const char *vfile, const char *pfile);
  void Compile(const char *vtext, const char *ftext);
  void Bind();
  void Unbind();
  void SetInputTexture(uint slot, const char *name, GLTexture *texture);
  void SetInputMatrix(const char *name, const mat4 &matrix);
  void SetFloat(const char *name, const float v);
  void SetInt(const char *name, const int v);
  void SetUInt(const char *name, const uint v);
  uint vertex = 0, pixel = 0, ID = 0;
};
#endif