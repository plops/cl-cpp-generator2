#ifndef VIS_00_BASE_H
#define VIS_00_BASE_H
#include "utils.h"
;
#include "globals.h"
;
#include <iostream>
#include <chrono>
#include <thread>
#include <typeinfo>
;
#include <include/gpu/GrBackendSurface.h>
#include <include/gpu/GrDirectContext.h>
#include <SDL2/SDL.h>
#include <include/core/SkCanvas.h>
#include <include/core/SkGraphics.h>
#include <include/core/SkSurface.h>
#include <include/gpu/gl/GrGLInterface.h>
#include <src/gpu/gl/GrGLUtil.h>
#include <GL/gl.h>
;
// header;
void skia_init (SkiaGLPrivate& s, int w, int h)  ;  
int main (int argc, char** argv)  ;  
#endif