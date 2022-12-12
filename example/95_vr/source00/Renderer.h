#ifndef RENDERER_H
#define RENDERER_H

#include "bla.h"
class Renderer  {
        public:
        std::vector<Framebuffer> framebuffers;
        Program program;
        Geometry geometry;
         Renderer (GLsizei width, GLsizei height)     ;  
};

#endif /* !RENDERER_H */