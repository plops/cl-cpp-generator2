// no preamble
#include "AttribPointer.h"
#include "bah.h"
AttribPointer::AttribPointer(GLint size, GLenum type, GLboolean normalized,
                             GLsizei stride, const GLvoid *pointer)
    : size(size), type(type), normalized(normalized), stride(stride),
      pointer(pointer) {}