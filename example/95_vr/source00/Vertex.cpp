// no preamble
#include "Vertex.h"
#include "bah.h"
Vertex::Vertex(GLint size, GLenum type, GLboolean normalized, GLsizei stride,
               const GLvoid *pointer)
    : size(size), type(type), normalized(normalized), stride(stride),
      pointer(pointer) {}