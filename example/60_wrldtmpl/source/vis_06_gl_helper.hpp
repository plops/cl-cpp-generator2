#ifndef VIS_06_GL_HELPER_H
#define VIS_06_GL_HELPER_H
#include "utils.h"
;
#include "globals.h"
;
// header

#define CheckGL()                                                              \
  { _CheckGL(__FILE__, __LINE__); };

void _CheckGL(const char *f, int l);

GLuint CreateVBO(const GLfloat *data, const uint size);

void BindVBO(const uint idx, const uint N, cont GLuint id);

void CheckShader(GLuint shader, const char *vshader, const char *fshader);

void CheckProgram(GLuint id, const char *vshader, const char *fshader);

void DrawQuad();
#endif