#ifndef ATTRIBPOINTER_H
#define ATTRIBPOINTER_H

#include "bla.h"
class AttribPointer  {
        public:
        GLint size;
        GLenum type;
        GLboolean normalized;
        GLsizei stride;
        const GLvoid* pointer;
         AttribPointer (GLint size, GLenum type, GLboolean normalized, GLsizei stride, const GLvoid* pointer)     ;  
};

#endif /* !ATTRIBPOINTER_H */