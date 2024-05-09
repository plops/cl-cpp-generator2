#ifndef VIS_02_SURFACE95_H
#define VIS_02_SURFACE95_H
#include "utils.h"
;
#include "globals.h"
;
#include <chrono>
#include <iostream>
#include <thread>
;
#include <FreeImage.h>
;
#include <vis_03_memory.hpp>
;
// header

// http://freeimage.sourceforge.net

#include <FreeImage.h>
#include <vis_03_memory.hpp>  ;
class Surface {
  enum { OWNER = 1 };

public:
  Surface(int w, int h, uint *a_Buffer);
  Surface(int w, int h);
  Surface(const char *file);
  ~Surface();
  void InitCharSet();
  void SetChar(const char *c1, const char *c2, const char *c3, const char *c4,
               const char *c5);
  void Print(const char *tt, int x1, int y1, uint c);
  void Clear(uint c);
  void LoadImage(const char *file);
  void CopyTo(Surface *dst, int a_X, int a_Y);
  void Box(int x1, int y1, int x2, int y2, uint color);
  void Bar(int x1, int y1, int x2, int y2, uint color);
  uint *buffer;
  int width;
  int height;
};
#endif