#ifndef VERTEX_H
#define VERTEX_H

#include <array>
#include "bla.h"
class Vertex  {
        public:
        std::array<float,4> position;
        std::array<float,4> color;
         Vertex (GLint size, GLenum type, GLboolean normalized, GLsizei stride, const GLvoid* pointer)     ;  
};

#endif /* !VERTEX_H */