#ifndef EGL_H
#define EGL_H

#include "bla.h"
class Egl  {
        public:
        EGLDisplay display;
        EGLContext context;
        EGLSurface surface;
         Egl ()     ;  
};

#endif /* !EGL_H */