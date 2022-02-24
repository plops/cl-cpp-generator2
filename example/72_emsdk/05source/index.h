#ifndef INDEX_H
#define INDEX_H

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define GL_GLEXT_PROTOTYPES
#define EGL_EGLEXT_PROTOTYPES
#endif
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "flextGL.h"
#include <cmath>
#include <GL/glu.h>
#include <functional>
void reset_state ()    ;  
void draw_triangle ()    ;  
void main_loop ()    ;  
int main (int argc, char** argv)    ;  

#endif /* !INDEX_H */