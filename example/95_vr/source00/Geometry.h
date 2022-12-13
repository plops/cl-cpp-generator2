#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "bla.h"
class Geometry  {
        public:
        GLuint vertex_array, vertex_buffer, index_buffer;
        Cube cube;
         Geometry ()     ;  
         ~Geometry ()     ;  
};

#endif /* !GEOMETRY_H */